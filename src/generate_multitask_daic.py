import csv
import json
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR / "../data/daic_woz").resolve()
DATASETS_ROOT = Path(os.environ["AUDIOLLM_DATASETS_ROOT"]).resolve()
PREPROCESSED_DIR = DATASETS_ROOT / "DAIC-WOZ/preprocessed"

SPLIT_CONFIGS = {
    "train": {
        "dataset_path": PREPROCESSED_DIR / "train_audio_segments",
        "csv_file_path": PREPROCESSED_DIR / "train_preprocessing_summary.csv",
        "output_file": DATA_DIR / "train/daic_woz_multitask.jsonl",
    },
    "val": {
        "dataset_path": PREPROCESSED_DIR / "dev_audio_segments",
        "csv_file_path": PREPROCESSED_DIR / "dev_preprocessing_summary.csv",
        "output_file": DATA_DIR / "val/daic_woz_multitask.jsonl",
    },
}


for split, config in SPLIT_CONFIGS.items():
    dataset_path = config["dataset_path"]
    csv_file_path = config["csv_file_path"]
    output_file = config["output_file"]
    output_file.parent.mkdir(parents=True, exist_ok=True)

    participant_to_label = {}
    with open(csv_file_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            participant_id = row["Participant_ID"]
            phq8_binary = row["PHQ8_Binary"]

            if phq8_binary == "1":
                label = "抑郁"
            elif phq8_binary == "0":
                label = "非抑郁"
            else:
                print(f"Warning: unexpected PHQ8_Binary value for {participant_id}: {phq8_binary}")
                continue

            participant_to_label[participant_id] = label

    count = 0
    with open(output_file, "w", encoding="utf-8") as outfile:
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(".wav"):
                    subfolder_name = os.path.basename(root)

                    if subfolder_name in participant_to_label:
                        file_id = os.path.splitext(file)[0]
                        label = participant_to_label[subfolder_name]

                        json_obj = {
                            "key": file_id,
                            "task": f"{file_id}_抑郁症识别",
                            "target": label,
                        }

                        outfile.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
                        count += 1
                    else:
                        print(f"Warning: folder {subfolder_name} has no record in {csv_file_path.name}")

    print(f"JSONL ({split}) created: {output_file} | files: {count}")
