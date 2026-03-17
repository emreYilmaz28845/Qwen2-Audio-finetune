import os
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR / "../data/daic_woz").resolve()
QUERY_JSON_DIR = DATA_DIR / "query_json"
DATASETS_ROOT = Path(os.environ["AUDIOLLM_DATASETS_ROOT"]).resolve()
PREPROCESSED_DIR = DATASETS_ROOT / "DAIC-WOZ/preprocessed"

SPLIT_CONFIGS = {
    "train": {
        "dataset_path": PREPROCESSED_DIR / "train_audio_segments",
        "query_json_path": QUERY_JSON_DIR / "train_full_xcy_P.json",
        "output_file": DATA_DIR / "train/daic_woz_multiprompt.jsonl",
    },
    "val": {
        "dataset_path": PREPROCESSED_DIR / "dev_audio_segments",
        "query_json_path": QUERY_JSON_DIR / "val_full_xcy_P.json",
        "output_file": DATA_DIR / "val/daic_woz_multiprompt.jsonl",
    },
}

for split, config in SPLIT_CONFIGS.items():
    dataset_path = config["dataset_path"]
    query_json_path = config["query_json_path"]
    output_file = config["output_file"]
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Read the JSON file containing query and response
    with open(query_json_path, "r", encoding="utf-8") as f:
        query_data = json.load(f)

    # Create a mapping from file_id to query content
    file_id_to_query = {}

    for item in query_data:
        # Extract the file name from the audio tag
        audio_tag = item["query"].split("<audio>")[1].split("</audio>")[0]
        file_id = os.path.splitext(os.path.basename(audio_tag))[0]

        # Extract the content in the query other than the audio tag
        prompt_content = item["query"].split("</audio>")[1].strip()

        file_id_to_query[file_id] = prompt_content

    # Generate the JSONL file
    count = 0
    with open(output_file, "w", encoding="utf-8") as outfile:
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(".wav"):
                    file_id = os.path.splitext(file)[0]

                    if file_id in file_id_to_query:
                        json_obj = {
                            "task": f"{file_id}_抑郁症识别",
                            "prompt": f"<|audio_bos|><|AUDIO|><|audio_eos|>{file_id_to_query[file_id]}",
                        }

                        outfile.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
                        count += 1
                    else:
                        print(f"Warning: file {file_id} has no matching record in {query_json_path.name}")

    print(f"JSONL ({split}) created: {output_file} | files: {count}")
