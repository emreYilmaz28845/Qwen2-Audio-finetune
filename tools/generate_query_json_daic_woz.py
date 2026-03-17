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

PROMPT_TEXT = "请根据这段语音、其对应的文本转录和情感描述判断该说话人是抑郁还是非抑郁"

SPLIT_CONFIGS = {
    "train": {
        "csv_file": PREPROCESSED_DIR / "train_preprocessing_summary.csv",
        "secap_file": DATA_ROOT / "train" / "secap_metadata.jsonl",
        "output_file": QUERY_JSON_DIR / "train_full_xcy_P.json",
    },
    "val": {
        "csv_file": PREPROCESSED_DIR / "dev_preprocessing_summary.csv",
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


def load_transcripts(csv_file: Path) -> dict[str, dict[str, str]]:
    session_to_data: dict[str, dict[str, str]] = {}
    with open(csv_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            participant_id = row["Participant_ID"].strip()
            full_transcript = row["full_transcript"].strip()
            label = phq8_to_label(row["PHQ8_Binary"].strip())
            session_to_data[participant_id] = {
                "transcript": full_transcript,
                "response": label,
            }
    return session_to_data


def build_query(audio_path: str, emotion: str, transcript: str) -> str:
    emotion_text = str([emotion])
    return (
        f"Audio:<audio>{audio_path}</audio>\n"
        f"{PROMPT_TEXT}\n"
        f"情感描述: {emotion_text}\n"
        f"文本转录: {transcript}"
    )


def build_split(split: str, csv_file: Path, secap_file: Path, output_file: Path) -> None:
    session_to_data = load_transcripts(csv_file)
    rows = []

    for item in read_jsonl(secap_file):
        key = str(item.get("key", "")).strip()
        audio_path = str(item.get("audio_path", "")).strip()
        emotion = str(item.get("emotion", "")).strip()

        if not key or not audio_path or not emotion:
            continue

        segment_name = Path(audio_path).stem
        session_id = extract_session_id(segment_name)
        transcript_row = session_to_data.get(session_id)
        if transcript_row is None:
            print(f"Warning: no transcript found for session {session_id} ({split})")
            continue

        rows.append(
            {
                "query": build_query(
                    audio_path=audio_path,
                    emotion=emotion,
                    transcript=transcript_row["transcript"],
                ),
                "response": transcript_row["response"],
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
            csv_file=config["csv_file"],
            secap_file=config["secap_file"],
            output_file=config["output_file"],
        )


if __name__ == "__main__":
    main()
