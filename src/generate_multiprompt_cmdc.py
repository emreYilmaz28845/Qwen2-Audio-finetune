import argparse
import json
from pathlib import Path


PROMPT_TEMPLATE = (
    "<|audio_bos|><|AUDIO|><|audio_eos|>"
    "请根据这段语音、其对应的文本转录和情感描述判断该说话人是抑郁还是非抑郁\n"
    "情感描述: {emotion}\n"
    "文本转录: {transcript}"
)
TASK_SUFFIX = "_抑郁症识别"
EMOTION_MARKER = "情感描述:"


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def key_from_task(task):
    if not task.endswith(TASK_SUFFIX):
        raise ValueError(f"Unexpected task format: {task}")
    return task[: -len(TASK_SUFFIX)]


def extract_emotion_from_prompt(prompt, source_path):
    for line in prompt.splitlines():
        if line.startswith(EMOTION_MARKER):
            return line[len(EMOTION_MARKER):].strip()
    raise ValueError(f"Prompt in {source_path} is missing '{EMOTION_MARKER}'")


def build_emotion_lookup(paper_root):
    emotion_by_key = {}
    for fold_num in range(1, 6):
        fold_name = f"fold{fold_num}"
        for split in ("train", "test"):
            prompt_path = paper_root / fold_name / split / f"{fold_name}_multiprompt.jsonl"
            for item in read_jsonl(prompt_path):
                key = key_from_task(item["task"])
                emotion = extract_emotion_from_prompt(item["prompt"], prompt_path)
                emotion_by_key[key] = emotion
    return emotion_by_key


def read_transcript(dataset_root, key):
    subject, clip = key.split("_", 1)
    candidates = [clip]
    if clip.endswith("-"):
        candidates.append(clip[:-1])

    for candidate in candidates:
        transcript_path = dataset_root / subject / f"{candidate}.txt"
        if transcript_path.exists():
            return transcript_path.read_text(encoding="utf-8").strip()

    raise FileNotFoundError(
        f"Missing transcript for {key}: tried "
        + ", ".join(str(dataset_root / subject / f'{candidate}.txt') for candidate in candidates)
    )


def resolve_emotion(key, emotion_by_key):
    emotion = emotion_by_key.get(key)
    if emotion is not None:
        return emotion

    if key.endswith("-"):
        return emotion_by_key.get(key[:-1])

    return None


def build_prompt_items(multitask_items, emotion_by_key, dataset_root):
    items = []
    missing_emotions = []
    for item in multitask_items:
        key = item["key"]
        emotion = resolve_emotion(key, emotion_by_key)
        if emotion is None:
            missing_emotions.append(key)
            continue
        transcript = read_transcript(dataset_root, key)
        items.append(
            {
                "task": item["task"],
                "prompt": PROMPT_TEMPLATE.format(
                    emotion=emotion,
                    transcript=transcript,
                ),
            }
        )

    if missing_emotions:
        sample = ", ".join(missing_emotions[:5])
        raise ValueError(
            f"Missing paper emotion descriptions for {len(missing_emotions)} keys, e.g. {sample}"
        )

    return items


def generate_all_folds(cmdc_root, paper_root, dataset_root):
    emotion_by_key = build_emotion_lookup(paper_root)
    print(f"Loaded {len(emotion_by_key)} emotion descriptions from {paper_root}")

    for fold_num in range(1, 6):
        fold_name = f"fold{fold_num}"
        for split in ("train", "test"):
            split_dir = cmdc_root / fold_name / split
            multitask_path = split_dir / f"{fold_name}_multitask.jsonl"
            output_path = split_dir / f"{fold_name}_multiprompt.jsonl"

            multitask_items = read_jsonl(multitask_path)
            prompt_items = build_prompt_items(multitask_items, emotion_by_key, dataset_root)

            if len(prompt_items) != len(multitask_items):
                raise ValueError(
                    f"{fold_name}/{split} produced {len(prompt_items)} prompts for "
                    f"{len(multitask_items)} multitask items"
                )

            write_jsonl(output_path, prompt_items)
            print(f"{multitask_path} -> {output_path} ({len(prompt_items)} samples)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate CMDC multiprompt manifests using paper emotion descriptions and live CMDC transcripts."
    )
    parser.add_argument(
        "--cmdc_root",
        type=Path,
        default=(Path(__file__).resolve().parent / "../data/cmdc").resolve(),
        help="CMDC root containing fold*/train and fold*/test directories.",
    )
    parser.add_argument(
        "--paper_root",
        type=Path,
        default=(Path(__file__).resolve().parent / "../data/cmdc-paper").resolve(),
        help="Paper CMDC root containing fold*/train and fold*/test multiprompt manifests.",
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        required=True,
        help="Live CMDC dataset root containing subject folders like HC52/Q9.txt.",
    )
    args = parser.parse_args()

    generate_all_folds(
        cmdc_root=args.cmdc_root.resolve(),
        paper_root=args.paper_root.resolve(),
        dataset_root=args.dataset_root.resolve(),
    )
    print("All CMDC multiprompt manifests generated.")


if __name__ == "__main__":
    main()
