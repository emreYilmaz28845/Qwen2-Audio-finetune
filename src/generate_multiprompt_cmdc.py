import argparse
import json
from pathlib import Path


PROMPT_TEMPLATE = (
    "<|audio_bos|><|AUDIO|><|audio_eos|>"
    "请根据这段语音、其对应的文本转录和情感描述判断该说话人是抑郁还是非抑郁\n"
    "情感描述: {emotion}\n"
    "文本转录: {transcript}"
)


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_emotion_labels(label_file):
    emotion_by_key = {}
    with label_file.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split("\t")
            if len(parts) < 2:
                raise ValueError(f"{label_file}:{line_no} does not contain a tab-separated label")
            audio_path = Path(parts[0])
            dirname = audio_path.parent.name
            filename = audio_path.stem
            key = f"{dirname}_{filename}"
            emotion_by_key[key] = parts[1]
    return emotion_by_key


def read_transcript(text_root, key):
    subject, clip = key.split("_", 1)
    transcript_path = text_root / subject / f"{clip}.txt"
    if not transcript_path.exists():
        raise FileNotFoundError(f"Missing transcript for {key}: {transcript_path}")
    return transcript_path.read_text(encoding="utf-8").strip()


def build_prompt_items(multitask_items, emotion_by_key, text_root):
    items = []
    missing_emotions = []
    for item in multitask_items:
        key = item["key"]
        if key not in emotion_by_key:
            missing_emotions.append(key)
            continue
        transcript = read_transcript(text_root, key)
        items.append(
            {
                "task": item["task"],
                "prompt": PROMPT_TEMPLATE.format(
                    emotion=emotion_by_key[key],
                    transcript=transcript,
                ),
            }
        )

    if missing_emotions:
        sample = ", ".join(missing_emotions[:5])
        raise ValueError(f"Missing emotion labels for {len(missing_emotions)} keys, e.g. {sample}")

    return items


def generate_all_folds(cmdc_root, emotion_labels_dir, text_root):
    for fold_num in range(1, 6):
        fold_name = f"fold{fold_num}"
        for split in ("train", "test"):
            split_dir = cmdc_root / fold_name / split
            multitask_path = split_dir / f"{fold_name}_multitask.jsonl"
            label_path = emotion_labels_dir / f"{fold_name}_{split}_multiprompt.txt"
            output_path = split_dir / f"{fold_name}_multiprompt.jsonl"

            multitask_items = read_jsonl(multitask_path)
            emotion_by_key = read_emotion_labels(label_path)
            prompt_items = build_prompt_items(multitask_items, emotion_by_key, text_root)

            if len(prompt_items) != len(multitask_items):
                raise ValueError(
                    f"{fold_name}/{split} produced {len(prompt_items)} prompts for "
                    f"{len(multitask_items)} multitask items"
                )

            write_jsonl(output_path, prompt_items)
            print(f"{multitask_path} -> {output_path} ({len(prompt_items)} samples)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate CMDC multiprompt manifests in canonical multitask order."
    )
    parser.add_argument(
        "--cmdc_root",
        type=Path,
        default=(Path(__file__).resolve().parent / "../data/cmdc").resolve(),
        help="CMDC root containing fold*/train and fold*/test directories.",
    )
    parser.add_argument(
        "--emotion_labels_dir",
        type=Path,
        required=True,
        help="Directory containing fold*_train_multiprompt.txt and fold*_test_multiprompt.txt.",
    )
    parser.add_argument(
        "--text_root",
        type=Path,
        required=True,
        help="Root directory containing subject transcript folders like HC01/Q1.txt.",
    )
    args = parser.parse_args()

    generate_all_folds(
        cmdc_root=args.cmdc_root.resolve(),
        emotion_labels_dir=args.emotion_labels_dir.resolve(),
        text_root=args.text_root.resolve(),
    )
    print("All CMDC multiprompt manifests generated.")


if __name__ == "__main__":
    main()
