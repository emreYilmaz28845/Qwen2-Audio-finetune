import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


SPLITS = ("train", "val", "test")
DATASET_BASENAMES = {
    "cmdc": "cmdc",
    "daic_woz": "daic_woz",
    "eatd": "eatd",
    "merged": "merged",
}
TEXT_MARKER = "文本转录:"
MIN_TEXT_HASH_CHARS = 80
GROUPED_DATASET_NAMES = {"cmdc", "daic_woz", "eatd"}
_DAIC_PARTICIPANT_ID_PATTERN = re.compile(r"^(?P<participant_id>\d+)")
_EATD_SUBJECT_ID_PATTERN = re.compile(r"^(?P<subject_id>[^_]+)_")
_CMDC_SUBJECT_ID_PATTERN = re.compile(r"^(?P<subject_id>[^_]+)_Q\d+$")


@dataclass
class SplitIssue:
    severity: str
    message: str


class SplitValidationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_audio_path(path_text: str, split_dir: Path) -> str:
    path = Path(path_text)
    if path.is_absolute():
        return str(path.resolve())
    return str((split_dir / path).resolve())


def normalize_text_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def transcript_or_prompt_hash(prompt: str):
    if not prompt:
        return None
    text = prompt
    marker_index = prompt.find(TEXT_MARKER)
    if marker_index != -1:
        text = prompt[marker_index + len(TEXT_MARKER):]
    text = normalize_text_for_hash(text)
    if len(text) < MIN_TEXT_HASH_CHARS:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            item["_line_no"] = line_no
            records.append(item)
    return records


def read_scp(path: Path, split_dir: Path):
    entries = {}
    duplicates = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            key, audio_path = stripped.split(maxsplit=1)
            if key in entries:
                duplicates.append((key, line_no))
            entries[key] = normalize_audio_path(audio_path, split_dir)
    return entries, duplicates


def read_prompts(path: Path):
    if not path.exists():
        return {}
    prompts = {}
    for item in read_jsonl(path):
        task = item.get("task")
        if task:
            prompts[task] = item.get("prompt", "")
    return prompts


def resolve_group_id(dataset_name: str, item: dict) -> str:
    explicit_group = item.get("group_id") or item.get("participant_id")
    if explicit_group:
        return str(explicit_group)

    item_dataset = item.get("dataset") or dataset_name
    source_key = item.get("source_key") or item.get("key", "")
    if item_dataset in GROUPED_DATASET_NAMES:
        if item_dataset == "daic_woz":
            match = _DAIC_PARTICIPANT_ID_PATTERN.match(source_key)
        elif item_dataset == "eatd":
            match = _EATD_SUBJECT_ID_PATTERN.match(source_key)
        else:
            match = _CMDC_SUBJECT_ID_PATTERN.match(source_key)
        if match is None:
            raise ValueError(f"Could not extract group ID for dataset={item_dataset!r} key={source_key!r}.")
        return next(iter(match.groupdict().values()))
    return source_key


def split_paths(root: Path, split: str, basename: str):
    split_dir = root / split
    return {
        "dir": split_dir,
        "scp": split_dir / f"{basename}.scp",
        "task": split_dir / f"{basename}_multitask.jsonl",
        "prompt": split_dir / f"{basename}_multiprompt.jsonl",
    }


def load_split(root: Path, split: str, dataset_name: str, basename: str, issues):
    paths = split_paths(root, split, basename)
    missing = [name for name in ("scp", "task") if not paths[name].exists()]
    if missing:
        issues.append(SplitIssue("error", f"{dataset_name}/{split}: missing {', '.join(missing)} file(s)"))
        return []

    scp_entries, scp_duplicates = read_scp(paths["scp"], paths["dir"])
    for key, line_no in scp_duplicates:
        issues.append(SplitIssue("error", f"{dataset_name}/{split}: duplicate scp key {key!r} at line {line_no}"))

    task_records = read_jsonl(paths["task"])
    prompts = read_prompts(paths["prompt"])
    task_keys = [item.get("key") for item in task_records]
    for key, count in Counter(task_keys).items():
        if count > 1:
            issues.append(SplitIssue("error", f"{dataset_name}/{split}: duplicate task key {key!r} x{count}"))

    missing_scp = sorted(key for key in task_keys if key not in scp_entries)
    if missing_scp:
        preview = ", ".join(missing_scp[:10])
        suffix = " ..." if len(missing_scp) > 10 else ""
        issues.append(SplitIssue("error", f"{dataset_name}/{split}: {len(missing_scp)} task key(s) missing from scp: {preview}{suffix}"))

    extra_scp = sorted(key for key in scp_entries if key not in set(task_keys))
    if extra_scp:
        preview = ", ".join(extra_scp[:10])
        suffix = " ..." if len(extra_scp) > 10 else ""
        issues.append(SplitIssue("error", f"{dataset_name}/{split}: {len(extra_scp)} scp key(s) missing from tasks: {preview}{suffix}"))

    loaded = []
    for item in task_records:
        key = item.get("key")
        task = item.get("task")
        prompt = prompts.get(task, "")
        loaded.append(
            {
                "split": split,
                "key": key,
                "task": task,
                "target": item.get("target"),
                "dataset": item.get("dataset") or dataset_name,
                "source_key": item.get("source_key") or key,
                "group_id": resolve_group_id(dataset_name, item),
                "audio_path": scp_entries.get(key),
                "text_hash": transcript_or_prompt_hash(prompt),
                "line_no": item.get("_line_no"),
            }
        )
    return loaded


def summarize_records(records):
    participants_by_split = defaultdict(set)
    segments_by_split = Counter()
    labels_by_split = defaultdict(Counter)
    for record in records:
        split = record["split"]
        participants_by_split[split].add(record["group_id"])
        segments_by_split[split] += 1
        labels_by_split[split][record["target"]] += 1
    return {
        "segments_by_split": dict(segments_by_split),
        "participants_by_split": {split: len(values) for split, values in participants_by_split.items()},
        "labels_by_split": {split: dict(counts) for split, counts in labels_by_split.items()},
    }


def validate_loaded_records(dataset_name: str, records, issues, strict: bool = True):
    by_group = defaultdict(list)
    by_key = defaultdict(list)
    by_audio = defaultdict(list)
    by_text = defaultdict(list)

    for record in records:
        by_group[record["group_id"]].append(record)
        by_key[record["key"]].append(record)
        if record["audio_path"]:
            by_audio[record["audio_path"]].append(record)
        if record["text_hash"]:
            by_text[record["text_hash"]].append(record)

    for key, key_records in by_key.items():
        splits = {record["split"] for record in key_records}
        if len(splits) > 1:
            issues.append(SplitIssue("error", f"{dataset_name}: key {key!r} appears in multiple splits: {sorted(splits)}"))

    for group_id, group_records in by_group.items():
        splits = {record["split"] for record in group_records}
        labels = {record["target"] for record in group_records}
        if len(labels) > 1:
            issues.append(SplitIssue("error", f"{dataset_name}: group {group_id!r} has conflicting labels: {sorted(labels)}"))
        if len(splits) > 1:
            issues.append(SplitIssue("error", f"{dataset_name}: group {group_id!r} appears in multiple splits: {sorted(splits)}"))

    for audio_path, audio_records in by_audio.items():
        splits = {record["split"] for record in audio_records}
        if len(splits) > 1:
            keys = sorted({record["key"] for record in audio_records})
            issues.append(SplitIssue("error", f"{dataset_name}: audio path appears in multiple splits: {audio_path} keys={keys[:5]}"))

    if strict:
        for text_hash, text_records in by_text.items():
            splits = {record["split"] for record in text_records}
            groups = {record["group_id"] for record in text_records}
            if len(splits) > 1 and len(groups) > 1:
                keys = sorted({record["key"] for record in text_records})
                issues.append(SplitIssue("error", f"{dataset_name}: transcript/prompt hash {text_hash[:12]} appears across splits: keys={keys[:5]}"))


def validate_dataset_splits(dataset_name: str, root: str | Path, strict: bool = True, splits=SPLITS):
    dataset_name = dataset_name.strip().lower()
    if dataset_name not in DATASET_BASENAMES:
        raise ValueError(f"Unsupported dataset_name={dataset_name!r}. Expected one of {sorted(DATASET_BASENAMES)}.")

    root = Path(root)
    basename = DATASET_BASENAMES[dataset_name]
    issues = []
    records = []
    for split in splits:
        records.extend(load_split(root, split, dataset_name, basename, issues))

    validate_loaded_records(dataset_name, records, issues, strict=strict)
    errors = [issue for issue in issues if issue.severity == "error"]
    summary = summarize_records(records)
    summary["dataset_name"] = dataset_name
    summary["root"] = str(root)
    summary["strict"] = bool(strict)
    summary["issues"] = [issue.__dict__ for issue in issues]
    summary["ok"] = not errors
    if errors:
        message = "\n".join(f"- {issue.message}" for issue in errors)
        raise SplitValidationError(f"Split validation failed for {dataset_name}:\n{message}")
    return summary


def validate_all(project_root: str | Path = ".", strict: bool = True):
    project_root = Path(project_root)
    summaries = {}
    for dataset_name in ("daic_woz", "eatd", "cmdc", "merged"):
        summaries[dataset_name] = validate_dataset_splits(
            dataset_name,
            project_root / "data" / dataset_name,
            strict=strict,
        )
    return summaries


def cli_main(argv=None):
    parser = argparse.ArgumentParser(description="Validate clean train/val/test manifest splits.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--dataset", choices=sorted(DATASET_BASENAMES), help="Dataset to validate.")
    parser.add_argument("--root", type=Path, help="Dataset root override.")
    parser.add_argument("--all", action="store_true", help="Validate all primary datasets.")
    parser.add_argument("--strict", action="store_true", help="Fail on exact transcript/prompt hash overlap too.")
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON summary output path.")
    args = parser.parse_args(argv)

    if not args.all and not args.dataset:
        parser.error("Use --all or --dataset.")

    if args.all:
        summaries = validate_all(args.project_root, strict=args.strict)
    else:
        root = args.root or (args.project_root / "data" / args.dataset)
        summaries = {
            args.dataset: validate_dataset_splits(args.dataset, root, strict=args.strict),
        }

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
