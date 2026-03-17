import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR / "../data/daic_woz").resolve()
DATASETS_ROOT = Path(os.environ["AUDIOLLM_DATASETS_ROOT"]).resolve()
DATASETS_PREFIX = os.environ.get("AUDIOLLM_DATASETS_PREFIX", "../Datasets")
PREPROCESSED_DIR = DATASETS_ROOT / "DAIC-WOZ/preprocessed"

SPLIT_CONFIGS = {
    "train": {
        "dataset_path": PREPROCESSED_DIR / "train_audio_segments",
        "output_file": DATA_DIR / "train/daic_woz.scp",
    },
    "val": {
        "dataset_path": PREPROCESSED_DIR / "dev_audio_segments",
        "output_file": DATA_DIR / "val/daic_woz.scp",
    },
}


for split, config in SPLIT_CONFIGS.items():
    dataset_path = config["dataset_path"]
    output_file = config["output_file"]
    output_file.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(".wav"):
                    abs_path = Path(root) / file
                    rel_path = os.path.join(DATASETS_PREFIX, os.path.relpath(abs_path, start=DATASETS_ROOT))
                    file_id = os.path.splitext(file)[0]
                    f.write(f"{file_id} {rel_path}\n")
                    count += 1

    print(f"SCP ({split}) created: {output_file} | files: {count}")
