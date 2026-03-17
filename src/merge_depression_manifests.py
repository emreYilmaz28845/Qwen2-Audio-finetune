import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_SPLITS = {
    "cmdc": {"train": "train", "val": "test"},
    "eatd": {"train": "train", "val": "test"},
    "daic_woz": {"train": "train", "val": "val"},
}

DEFAULT_FILES = {
    "cmdc": {
        "scp": "{fold}.scp",
        "multitask": "{fold}_multitask.jsonl",
        "multiprompt": "{fold}_multiprompt.jsonl",
    },
    "eatd": {
        "scp": "eatd.scp",
        "multitask": "eatd_multitask.jsonl",
        "multiprompt": "eatd_multiprompt.jsonl",
    },
    "daic_woz": {
        "scp": "daic_woz.scp",
        "multitask": "daic_woz_multitask.jsonl",
        "multiprompt": "daic_woz_multiprompt.jsonl",
    },
}


def parse_args():
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]

    parser = argparse.ArgumentParser(
        description="Merge CMDC, EATD, and DAIC-WOZ manifests into one dataset folder."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root,
        help="Qwen2-Audio-finetune project root.",
    )
    parser.add_argument(
        "--cmdc-root",
        type=Path,
        default=project_root / "data/cmdc",
        help="CMDC data root containing fold directories.",
    )
    parser.add_argument(
        "--eatd-root",
        type=Path,
        default=project_root / "data/eatd",
        help="EATD data root containing train/test manifests.",
    )
    parser.add_argument(
        "--daic-root",
        type=Path,
        default=project_root / "data/daic_woz",
        help="DAIC-WOZ data root containing train/val manifests.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "data/merged",
        help="Output directory for merged manifests.",
    )
    parser.add_argument(
        "--cmdc-fold",
        default="fold1",
        help="CMDC fold to use. A single fold is used to avoid cross-fold leakage.",
    )
    return parser.parse_args()


def read_scp(scp_path: Path):
    entries = {}
    with scp_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, wav_path = line.split(maxsplit=1)
            entries[key] = wav_path
    return entries


def read_jsonl(jsonl_path: Path):
    records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def resolve_audio_path(raw_path: str, split_dir: Path, project_root: Path) -> str:
    raw = Path(raw_path)
    if raw.is_absolute():
        return str(raw)

    candidates = [
        (split_dir / raw).resolve(),
        (project_root / raw).resolve(),
        (project_root.parent / raw).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return raw_path


def remap_task_name(old_task: str, old_key: str, new_key: str) -> str:
    prefix = f"{old_key}_"
    if old_task.startswith(prefix):
        return f"{new_key}_{old_task[len(prefix):]}"
    return f"{new_key}_抑郁症识别"


def load_source_split(dataset_name: str, split_name: str, root: Path, project_root: Path, fold: str = ""):
    source_split_name = DEFAULT_SPLITS[dataset_name][split_name]
    split_dir = root / fold / source_split_name if dataset_name == "cmdc" else root / source_split_name
    file_cfg = DEFAULT_FILES[dataset_name]

    scp_path = split_dir / file_cfg["scp"].format(fold=fold)
    multitask_path = split_dir / file_cfg["multitask"].format(fold=fold)
    multiprompt_path = split_dir / file_cfg["multiprompt"].format(fold=fold)

    missing = [str(path) for path in (scp_path, multitask_path, multiprompt_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files for {dataset_name}/{split_name}: {missing}")

    scp_entries = read_scp(scp_path)
    multitask_records = read_jsonl(multitask_path)
    prompt_lookup = {item["task"]: item["prompt"] for item in read_jsonl(multiprompt_path)}

    merged_records = []
    merged_prompts = []
    seen_tasks = set()
    skipped_keys = []

    for item in multitask_records:
        old_key = item.get("key")
        old_task = item.get("task")
        if old_key not in scp_entries or old_task not in prompt_lookup:
            skipped_keys.append(old_key or old_task or "<missing>")
            continue

        new_key = f"{dataset_name}__{old_key}"
        new_task = remap_task_name(old_task, old_key, new_key)
        merged_records.append(
            {
                "key": new_key,
                "task": new_task,
                "target": item["target"],
                "dataset": dataset_name,
                "source_split": source_split_name,
                "source_key": old_key,
                "audio_path": resolve_audio_path(scp_entries[old_key], split_dir, project_root),
            }
        )

        if new_task not in seen_tasks:
            merged_prompts.append({"task": new_task, "prompt": prompt_lookup[old_task]})
            seen_tasks.add(new_task)

    return {
        "records": merged_records,
        "prompts": merged_prompts,
        "skipped": skipped_keys,
        "source_split": source_split_name,
    }


def write_split(output_dir: Path, records, prompts):
    output_dir.mkdir(parents=True, exist_ok=True)
    scp_path = output_dir / "merged.scp"
    multitask_path = output_dir / "merged_multitask.jsonl"
    multiprompt_path = output_dir / "merged_multiprompt.jsonl"

    records = sorted(records, key=lambda item: item["key"])
    prompts = sorted(prompts, key=lambda item: item["task"])

    with scp_path.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(f"{item['key']} {item['audio_path']}\n")

    with multitask_path.open("w", encoding="utf-8") as f:
        for item in records:
            payload = {
                "key": item["key"],
                "task": item["task"],
                "target": item["target"],
                "dataset": item["dataset"],
                "source_split": item["source_split"],
                "source_key": item["source_key"],
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    with multiprompt_path.open("w", encoding="utf-8") as f:
        for item in prompts:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    project_root = args.project_root.resolve()

    roots = {
        "cmdc": args.cmdc_root.resolve(),
        "eatd": args.eatd_root.resolve(),
        "daic_woz": args.daic_root.resolve(),
    }

    merged = defaultdict(lambda: {"records": [], "prompts": [], "skipped": []})
    for split_name in ("train", "val"):
        for dataset_name, root in roots.items():
            fold = args.cmdc_fold if dataset_name == "cmdc" else ""
            result = load_source_split(dataset_name, split_name, root, project_root, fold=fold)
            merged[split_name]["records"].extend(result["records"])
            merged[split_name]["prompts"].extend(result["prompts"])
            merged[split_name]["skipped"].extend(result["skipped"])

    output_root = args.output_root.resolve()
    write_split(output_root / "train", merged["train"]["records"], merged["train"]["prompts"])
    write_split(output_root / "val", merged["val"]["records"], merged["val"]["prompts"])

    stats = {
        "cmdc_fold": args.cmdc_fold,
        "train_total": len(merged["train"]["records"]),
        "val_total": len(merged["val"]["records"]),
        "train_by_dataset": dict(Counter(item["dataset"] for item in merged["train"]["records"])),
        "val_by_dataset": dict(Counter(item["dataset"] for item in merged["val"]["records"])),
        "train_by_target": dict(Counter(item["target"] for item in merged["train"]["records"])),
        "val_by_target": dict(Counter(item["target"] for item in merged["val"]["records"])),
        "skipped_records": {
            "train": merged["train"]["skipped"][:20],
            "val": merged["val"]["skipped"][:20],
        },
    }
    (output_root / "merge_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Merged manifests created")
    print(f"output: {output_root}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
