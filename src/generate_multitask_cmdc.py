import argparse
import json
from pathlib import Path


def read_scp_keys(scp_path):
    keys = []
    with scp_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 2:
                raise ValueError(f"{scp_path}:{line_no} is not a valid SCP entry")
            keys.append(parts[0])
    return keys


def build_multitask_items(keys):
    items = []
    for key in keys:
        target = "抑郁" if key.startswith("MDD") else "非抑郁"
        items.append(
            {
                "key": key,
                "task": f"{key}_抑郁症识别",
                "target": target,
            }
        )
    return items


def write_jsonl(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def generate_from_scp_root(scp_root):
    for fold_num in range(1, 6):
        fold_name = f"fold{fold_num}"
        for split in ("train", "test"):
            split_dir = scp_root / fold_name / split
            scp_path = split_dir / f"{fold_name}.scp"
            output_path = split_dir / f"{fold_name}_multitask.jsonl"
            keys = read_scp_keys(scp_path)
            items = build_multitask_items(keys)
            write_jsonl(output_path, items)
            print(f"{scp_path} -> {output_path} ({len(items)} samples)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate CMDC multitask manifests directly from canonical fold SCP files."
    )
    parser.add_argument(
        "--scp_root",
        type=Path,
        default=(Path(__file__).resolve().parent / "../data/cmdc").resolve(),
        help="CMDC root containing fold*/train and fold*/test directories.",
    )
    args = parser.parse_args()

    scp_root = args.scp_root.resolve()
    generate_from_scp_root(scp_root)
    print("All CMDC multitask manifests generated.")


if __name__ == "__main__":
    main()
