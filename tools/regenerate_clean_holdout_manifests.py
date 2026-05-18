#!/usr/bin/env python3
import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_MODES = ("full", "audiotext", "textonly")
SPLITS = ("train", "val", "test")
AUDIO_TOKENS = "<|audio_bos|><|AUDIO|><|audio_eos|>"


DATASET_BASENAMES = {
    "cmdc": "cmdc",
    "daic_woz": "daic_woz",
    "eatd": "eatd",
}


SOURCE_CONFIG = {
    "daic_woz": {
        "legacy_splits": ("train", "val"),
        "legacy_basename": "daic_woz",
        "note": "Clean train/val/test is derived from the labeled DAIC-WOZ train+dev participants because official test labels/audio are not available in this repo.",
    },
    "eatd": {
        "legacy_splits": ("train", "test"),
        "legacy_basename": "eatd",
        "note": "EATD source train/test folder names are preserved in participant_id to avoid numeric ID collisions.",
    },
    "cmdc": {
        "legacy_splits": (),
        "legacy_basename": "",
        "note": "CMDC clean holdout is derived from the union of legacy folds; old fold directories are retained only for audit/secondary diagnostics.",
    },
}


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_scp(path: Path):
    entries = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            key, audio_path = stripped.split(maxsplit=1)
            entries[key] = audio_path
    return entries


def read_prompts(split_dir: Path, basename: str):
    suffix_by_mode = {
        "full": "multiprompt",
        "audiotext": "multiprompt_audiotext",
        "textonly": "multiprompt_textonly",
    }
    prompts = {}
    for mode, suffix in suffix_by_mode.items():
        path = split_dir / f"{basename}_{suffix}.jsonl"
        mode_prompts = {}
        if path.exists():
            for item in read_jsonl(path):
                mode_prompts[item["task"]] = item.get("prompt", "")
        prompts[mode] = mode_prompts
    return prompts


def prompt_for_mode(prompt_lookup, task: str, mode: str):
    if task in prompt_lookup.get(mode, {}):
        return prompt_lookup[mode][task]
    return prompt_lookup.get("full", {}).get(task, "")


def cm_metadata(source_key: str):
    participant = source_key.split("_", 1)[0]
    return participant, source_key


def daic_metadata(source_key: str):
    match = re.match(r"^(\d+)", source_key)
    if not match:
        raise ValueError(f"Could not extract DAIC participant from key={source_key!r}")
    return match.group(1), source_key


def infer_eatd_source_split(raw_split: str, audio_path: str):
    normalized = audio_path.replace("\\", "/").lower()
    if "/validation/" in normalized:
        return "validation"
    if "/train/" in normalized:
        return "train"
    return raw_split


def eatd_metadata(source_key: str, raw_split: str, audio_path: str):
    subject, segment = source_key.split("_", 1)
    source_split = infer_eatd_source_split(raw_split, audio_path)
    participant = f"{source_split}_{subject}"
    output_key = f"{participant}_{segment}"
    return participant, output_key


def remap_task(old_task: str, old_key: str, new_key: str):
    prefix = f"{old_key}_"
    if old_task.startswith(prefix):
        return f"{new_key}_{old_task[len(prefix):]}"
    return f"{new_key}_抑郁症识别"


def load_legacy_split(dataset_name: str, root: Path, split: str, basename: str):
    split_dir = root / split
    scp_path = split_dir / f"{basename}.scp"
    task_path = split_dir / f"{basename}_multitask.jsonl"
    if not scp_path.exists() or not task_path.exists():
        raise FileNotFoundError(f"Missing legacy source files for {dataset_name}/{split}: {scp_path}, {task_path}")

    scp = read_scp(scp_path)
    prompts = read_prompts(split_dir, basename)
    records = []
    for item in read_jsonl(task_path):
        source_key = item["key"]
        if source_key not in scp:
            continue
        audio_path = scp[source_key]
        if dataset_name == "daic_woz":
            participant_id, output_key = daic_metadata(source_key)
            source_split = split
        elif dataset_name == "eatd":
            participant_id, output_key = eatd_metadata(source_key, split, audio_path)
            source_split = participant_id.rsplit("_", 1)[0]
        else:
            participant_id, output_key = cm_metadata(source_key)
            source_split = split

        task = remap_task(item["task"], source_key, output_key)
        records.append(
            {
                "dataset": dataset_name,
                "source_split": source_split,
                "source_key": source_key,
                "key": output_key,
                "task": task,
                "target": item["target"],
                "participant_id": participant_id,
                "group_id": participant_id,
                "audio_path": audio_path,
                "prompts": {
                    mode: prompt_for_mode(prompts, item["task"], mode)
                    for mode in PROMPT_MODES
                },
            }
        )
    return records


def load_cmdc_legacy(root: Path):
    records_by_key = {}
    conflicts = []
    for fold in range(1, 6):
        fold_name = f"fold{fold}"
        for split in ("train", "test"):
            split_dir = root / fold_name / split
            basename = fold_name
            if not split_dir.exists():
                continue
            for record in load_legacy_split("cmdc", root / fold_name, split, basename):
                existing = records_by_key.get(record["source_key"])
                if existing is None:
                    record["legacy_sources"] = [f"{fold_name}/{split}"]
                    records_by_key[record["source_key"]] = record
                    continue
                existing["legacy_sources"].append(f"{fold_name}/{split}")
                if existing["target"] != record["target"] or existing["audio_path"] != record["audio_path"]:
                    conflicts.append((record["source_key"], existing["legacy_sources"], f"{fold_name}/{split}"))
    if conflicts:
        preview = ", ".join(item[0] for item in conflicts[:10])
        raise RuntimeError(f"CMDC legacy fold conflicts detected for keys: {preview}")
    return list(records_by_key.values())


def source_cache_path(cache_root: Path, dataset_name: str):
    return cache_root / f"{dataset_name}_source_records.jsonl"


def load_source_records(dataset_name: str, data_root: Path, cache_root: Path, refresh_source: bool):
    cache_path = source_cache_path(cache_root, dataset_name)
    if cache_path.exists() and not refresh_source:
        return read_jsonl(cache_path)

    cfg = SOURCE_CONFIG[dataset_name]
    if dataset_name == "cmdc":
        records = load_cmdc_legacy(data_root / dataset_name)
    else:
        records = []
        for split in cfg["legacy_splits"]:
            records.extend(
                load_legacy_split(
                    dataset_name,
                    data_root / dataset_name,
                    split,
                    cfg["legacy_basename"],
                )
            )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(cache_path, sorted(records, key=lambda item: item["key"]))
    return records


def split_participants(records, seed: int, train_ratio: float, val_ratio: float):
    labels_by_participant = defaultdict(set)
    for record in records:
        labels_by_participant[record["participant_id"]].add(record["target"])
    conflicts = {key: sorted(values) for key, values in labels_by_participant.items() if len(values) != 1}
    if conflicts:
        raise RuntimeError(f"Conflicting labels by participant: {conflicts}")

    by_label = defaultdict(list)
    for participant, labels in labels_by_participant.items():
        by_label[next(iter(labels))].append(participant)

    assignments = {}
    rng = random.Random(seed)
    for label, participants in sorted(by_label.items()):
        participants = sorted(participants)
        rng.shuffle(participants)
        count = len(participants)
        test_count = max(1, round(count * (1.0 - train_ratio - val_ratio))) if count >= 3 else 0
        val_count = max(1, round(count * val_ratio)) if count >= 3 else 0
        if test_count + val_count >= count:
            test_count = 1 if count >= 3 else 0
            val_count = 1 if count >= 3 else 0
        test_participants = participants[:test_count]
        val_participants = participants[test_count:test_count + val_count]
        train_participants = participants[test_count + val_count:]
        for participant in train_participants:
            assignments[participant] = "train"
        for participant in val_participants:
            assignments[participant] = "val"
        for participant in test_participants:
            assignments[participant] = "test"
    return assignments


def file_hash(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_dataset(dataset_name: str, records, output_root: Path, assignments, source_cache: Path, seed: int):
    basename = DATASET_BASENAMES[dataset_name]
    by_split = defaultdict(list)
    for record in records:
        split = assignments[record["participant_id"]]
        by_split[split].append(record)

    output_hashes = {}
    for split in SPLITS:
        split_dir = output_root / dataset_name / split
        split_dir.mkdir(parents=True, exist_ok=True)
        split_records = sorted(by_split[split], key=lambda item: item["key"])

        scp_path = split_dir / f"{basename}.scp"
        with scp_path.open("w", encoding="utf-8") as handle:
            for record in split_records:
                handle.write(f"{record['key']} {record['audio_path']}\n")

        task_rows = [
            {
                "key": record["key"],
                "task": record["task"],
                "target": record["target"],
                "dataset": dataset_name,
                "source_split": record["source_split"],
                "source_key": record["source_key"],
                "participant_id": record["participant_id"],
                "group_id": record["group_id"],
            }
            for record in split_records
        ]
        multitask_path = split_dir / f"{basename}_multitask.jsonl"
        write_jsonl(multitask_path, task_rows)
        write_jsonl(split_dir / f"{basename}_multitask_filtered.jsonl", task_rows)

        for mode in PROMPT_MODES:
            suffix = "multiprompt" if mode == "full" else f"multiprompt_{mode}"
            prompt_rows = [
                {
                    "task": record["task"],
                    "prompt": record["prompts"].get(mode) or record["prompts"].get("full", ""),
                }
                for record in split_records
            ]
            write_jsonl(split_dir / f"{basename}_{suffix}.jsonl", prompt_rows)

        output_hashes[split] = {
            "scp": file_hash(scp_path),
            "multitask": file_hash(multitask_path),
        }

    audit = build_audit(dataset_name, records, by_split, source_cache, output_hashes, seed)
    audit_path = output_root / dataset_name / "split_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit


def build_audit(dataset_name: str, records, by_split, source_cache: Path, output_hashes, seed: int):
    participant_labels = {}
    for record in records:
        participant_labels[record["participant_id"]] = record["target"]
    split_participants = {
        split: sorted({record["participant_id"] for record in split_records})
        for split, split_records in by_split.items()
    }
    return {
        "dataset_name": dataset_name,
        "seed": seed,
        "source_cache": str(source_cache),
        "note": SOURCE_CONFIG[dataset_name]["note"],
        "total_segments": len(records),
        "total_participants": len(participant_labels),
        "segments_by_split": {split: len(by_split[split]) for split in SPLITS},
        "participants_by_split": {split: len(split_participants.get(split, [])) for split in SPLITS},
        "labels_by_split": {
            split: dict(Counter(record["target"] for record in by_split[split]))
            for split in SPLITS
        },
        "participant_labels_by_split": {
            split: dict(Counter(participant_labels[participant] for participant in split_participants.get(split, [])))
            for split in SPLITS
        },
        "output_hashes": output_hashes,
    }


def remap_merged_task(old_task: str, old_key: str, new_key: str):
    prefix = f"{old_key}_"
    if old_task.startswith(prefix):
        return f"{new_key}_{old_task[len(prefix):]}"
    return f"{new_key}_抑郁症识别"


def write_merged(output_root: Path):
    merged_by_split = defaultdict(list)
    for dataset_name, basename in DATASET_BASENAMES.items():
        for split in SPLITS:
            split_dir = output_root / dataset_name / split
            scp = read_scp(split_dir / f"{basename}.scp")
            prompts_by_mode = read_prompts(split_dir, basename)
            for item in read_jsonl(split_dir / f"{basename}_multitask.jsonl"):
                old_key = item["key"]
                new_key = f"{dataset_name}__{old_key}"
                new_task = remap_merged_task(item["task"], old_key, new_key)
                merged_by_split[split].append(
                    {
                        "key": new_key,
                        "task": new_task,
                        "target": item["target"],
                        "dataset": dataset_name,
                        "source_split": split,
                        "source_key": old_key,
                        "participant_id": item.get("participant_id"),
                        "group_id": item.get("group_id"),
                        "audio_path": scp[old_key],
                        "prompts": {
                            mode: prompts_by_mode.get(mode, {}).get(item["task"], "")
                            for mode in PROMPT_MODES
                        },
                    }
                )

    output_hashes = {}
    for split in SPLITS:
        split_dir = output_root / "merged" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        records = sorted(merged_by_split[split], key=lambda item: item["key"])
        scp_path = split_dir / "merged.scp"
        with scp_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(f"{record['key']} {record['audio_path']}\n")
        task_rows = [
            {
                "key": record["key"],
                "task": record["task"],
                "target": record["target"],
                "dataset": record["dataset"],
                "source_split": record["source_split"],
                "source_key": record["source_key"],
                "participant_id": record["participant_id"],
                "group_id": record["group_id"],
            }
            for record in records
        ]
        multitask_path = split_dir / "merged_multitask.jsonl"
        write_jsonl(multitask_path, task_rows)
        write_jsonl(split_dir / "merged_multitask_filtered.jsonl", task_rows)
        for mode in PROMPT_MODES:
            suffix = "multiprompt" if mode == "full" else f"multiprompt_{mode}"
            write_jsonl(
                split_dir / f"merged_{suffix}.jsonl",
                [
                    {
                        "task": record["task"],
                        "prompt": record["prompts"].get(mode) or record["prompts"].get("full", ""),
                    }
                    for record in records
                ],
            )
        output_hashes[split] = {
            "scp": file_hash(scp_path),
            "multitask": file_hash(multitask_path),
        }

    audit = {
        "dataset_name": "merged",
        "source_datasets": sorted(DATASET_BASENAMES),
        "segments_by_split": {split: len(merged_by_split[split]) for split in SPLITS},
        "labels_by_split": {
            split: dict(Counter(record["target"] for record in merged_by_split[split]))
            for split in SPLITS
        },
        "by_source_dataset": {
            split: dict(Counter(record["dataset"] for record in merged_by_split[split]))
            for split in SPLITS
        },
        "output_hashes": output_hashes,
    }
    (output_root / "merged" / "split_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return audit


def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate clean participant-level train/val/test manifests in place.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--refresh-source", action="store_true", help="Rebuild source cache from current legacy manifests.")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = args.project_root.resolve()
    data_root = (args.data_root or project_root / "data").resolve()
    cache_root = data_root / "source_manifests" / "clean_holdout"

    audits = {}
    for dataset_name in ("daic_woz", "eatd", "cmdc"):
        records = load_source_records(dataset_name, data_root, cache_root, args.refresh_source)
        assignments = split_participants(
            records,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        audits[dataset_name] = write_dataset(
            dataset_name,
            records,
            data_root,
            assignments,
            source_cache_path(cache_root, dataset_name),
            args.seed,
        )

    audits["merged"] = write_merged(data_root)
    audit_path = data_root / "clean_holdout_manifest_audit.json"
    audit_path.write_text(json.dumps(audits, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audits, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
