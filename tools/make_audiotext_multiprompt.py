#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


AUDIO_TEXT_HEADER = "请根据这段语音及其对应的文本转录判断该说话人是抑郁还是非抑郁"
FULL_HEADER = "请根据这段语音、其对应的文本转录和情感描述判断该说话人是抑郁还是非抑郁"
EMOTION_MARKER = "情感描述:"
TRANSCRIPT_MARKER = "文本转录:"


def parse_args():
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]

    parser = argparse.ArgumentParser(
        description="Convert multiprompt JSONL files into the audio+text prompt template."
    )
    parser.add_argument("inputs", nargs="*", type=Path, help="Input multiprompt JSONL file(s).")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path when converting a single input file.",
    )
    parser.add_argument(
        "--suffix",
        default="_audiotext",
        help="Suffix to append before the file extension for batch conversion.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file(s) instead of creating a new file.",
    )
    parser.add_argument(
        "--cmdc-root",
        type=Path,
        default=project_root / "data/cmdc",
        help="CMDC root containing fold*/train and fold*/test directories.",
    )
    parser.add_argument(
        "--generate-cmdc-5fold",
        action="store_true",
        help="Generate audio+text files for all CMDC fold train/test multiprompt files.",
    )
    return parser.parse_args()


def build_output_path(input_path: Path, suffix: str) -> Path:
    return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")


def convert_prompt(prompt: str) -> str:
    lines = [line.strip() for line in prompt.splitlines() if line.strip()]

    if not any(TRANSCRIPT_MARKER in line for line in lines):
        raise ValueError("Prompt is missing the transcript marker.")

    converted_lines = []
    for line in lines:
        if FULL_HEADER in line:
            converted_lines.append(line.replace(FULL_HEADER, AUDIO_TEXT_HEADER, 1))
            continue
        if line.startswith(EMOTION_MARKER):
            continue
        converted_lines.append(line)

    return "\n".join(converted_lines)


def convert_file(input_path: Path, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            item = json.loads(stripped)
            if "prompt" not in item:
                raise KeyError(f"{input_path}:{line_no} is missing the 'prompt' field")

            item["prompt"] = convert_prompt(item["prompt"])
            dst.write(json.dumps(item, ensure_ascii=False) + "\n")


def collect_cmdc_inputs(cmdc_root: Path):
    inputs = []
    for fold in range(1, 6):
        for split in ("train", "test"):
            candidate = cmdc_root / f"fold{fold}" / split / f"fold{fold}_multiprompt.jsonl"
            if not candidate.exists():
                raise FileNotFoundError(f"Missing CMDC multiprompt file: {candidate}")
            inputs.append(candidate)
    return inputs


def resolve_jobs(args):
    inputs = list(args.inputs)
    if args.generate_cmdc_5fold:
        inputs.extend(collect_cmdc_inputs(args.cmdc_root.resolve()))

    unique_inputs = []
    seen = set()
    for path in inputs:
        resolved = path.resolve()
        if resolved not in seen:
            unique_inputs.append(resolved)
            seen.add(resolved)

    if not unique_inputs:
        raise ValueError("No input files were provided.")

    if args.output and len(unique_inputs) != 1:
        raise ValueError("--output can only be used with a single input file.")

    jobs = []
    for input_path in unique_inputs:
        if not input_path.exists():
            raise FileNotFoundError(f"Input file does not exist: {input_path}")

        if args.in_place:
            output_path = input_path
        elif args.output:
            output_path = args.output.resolve()
        else:
            output_path = build_output_path(input_path, args.suffix)

        jobs.append((input_path, output_path))
    return jobs


def main():
    args = parse_args()
    jobs = resolve_jobs(args)

    for input_path, output_path in jobs:
        convert_file(input_path, output_path)
        print(f"{input_path} -> {output_path}")


if __name__ == "__main__":
    main()
