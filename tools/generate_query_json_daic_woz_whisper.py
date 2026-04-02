#!/usr/bin/env python3
import csv
import json
import os
import re
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "daic_woz"
QUERY_JSON_DIR = DATA_ROOT / "query_json"
DATASETS_ROOT = Path(
    os.environ.get("AUDIOLLM_DATASETS_ROOT", "/media/emre/Backup/AudioLLM/Datasets")
).resolve()
PREPROCESSED_DIR = DATASETS_ROOT / "DAIC-WOZ" / "preprocessed"
WHISPER_TRANSCRIPTS_JSONL = DATASETS_ROOT / "DAIC-WOZ" / "whisper_transcripts.jsonl"

PROMPT_TEXT = "请根据这段语音、其对应的文本转录和情感描述判断该说话人是抑郁还是非抑郁"

SPLIT_CONFIGS = {
    "train": {
        "label_csv_file": PREPROCESSED_DIR / "train_preprocessing_summary.csv",
        "secap_file": DATA_ROOT / "train" / "secap_metadata.jsonl",
        "output_file": QUERY_JSON_DIR / "train_full_xcy_P.json",
    },
    "val": {
        "label_csv_file": PREPROCESSED_DIR / "dev_preprocessing_summary.csv",
        "secap_file": DATA_ROOT / "val" / "secap_metadata.jsonl",
        "output_file": QUERY_JSON_DIR / "val_full_xcy_P.json",
    },
}


def read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def phq8_to_label(value: str) -> str:
    if value == "1":
        return "抑郁"
    if value == "0":
        return "非抑郁"
    raise ValueError(f"Unexpected PHQ8_Binary value: {value}")


def extract_session_id(segment_name: str) -> str:
    match = re.match(r"(?P<session>\d+)_", segment_name)
    if not match:
        raise ValueError(f"Could not extract session id from segment name: {segment_name}")
    return match.group("session")


def load_labels(csv_file: Path) -> dict[str, str]:
    session_to_label: dict[str, str] = {}
    with open(csv_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            participant_id = row["Participant_ID"].strip()
            label = phq8_to_label(row["PHQ8_Binary"].strip())
            session_to_label[participant_id] = label
    return session_to_label


def normalize_audio_ref(audio_path: str) -> str:
    text = audio_path.strip().replace("\\", "/")
    marker = "DAIC-WOZ/"
    idx = text.find(marker)
    if idx >= 0:
        return text[idx:]
    return text


def load_transcripts(transcript_jsonl: Path) -> dict[str, str]:
    audio_to_transcript: dict[str, str] = {}
    for row in read_jsonl(transcript_jsonl):
        audio_path = str(row.get("audio_path", "")).strip()
        transcript = str(row.get("transcript", "")).strip()
        if not audio_path or not transcript:
            continue
        audio_to_transcript[normalize_audio_ref(audio_path)] = transcript
    return audio_to_transcript


def build_query(audio_path: str, emotion: str, transcript: str) -> str:
    emotion_text = str([emotion])
    return (
        f"Audio:<audio>{audio_path}</audio>\n"
        f"{PROMPT_TEXT}\n"
        f"情感描述: {emotion_text}\n"
        f"文本转录: {transcript}"
    )


def build_split(
    split: str,
    label_csv_file: Path,
    transcript_jsonl: Path,
    secap_file: Path,
    output_file: Path,
) -> None:
    session_to_label = load_labels(label_csv_file)
    audio_to_transcript = load_transcripts(transcript_jsonl)
    rows = []

    for item in read_jsonl(secap_file):
        key = str(item.get("key", "")).strip()
        audio_path = str(item.get("audio_path", "")).strip()
        emotion = str(item.get("emotion", "")).strip()

        if not key or not audio_path or not emotion:
            continue

        segment_name = Path(audio_path).stem
        session_id = extract_session_id(segment_name)
        transcript = audio_to_transcript.get(normalize_audio_ref(audio_path))
        if transcript is None:
            print(f"Warning: no transcript found for audio {audio_path} ({split})")
            continue
        label = session_to_label.get(session_id)
        if label is None:
            print(f"Warning: no label found for session {session_id} ({split})")
            continue

        rows.append(
            {
                "query": build_query(
                    audio_path=audio_path,
                    emotion=emotion,
                    transcript=transcript,
                ),
                "response": label,
            }
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"Query JSON ({split}) created: {output_file} | rows: {len(rows)}")


def main():
    for split, config in SPLIT_CONFIGS.items():
        build_split(
            split=split,
            label_csv_file=config["label_csv_file"],
            transcript_jsonl=WHISPER_TRANSCRIPTS_JSONL,
            secap_file=config["secap_file"],
            output_file=config["output_file"],
        )


if __name__ == "__main__":
    main()
