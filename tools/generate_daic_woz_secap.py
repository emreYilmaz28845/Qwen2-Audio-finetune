#!/usr/bin/env python3
import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DATASETS_ROOT = Path(
    os.environ.get("AUDIOLLM_DATASETS_ROOT", "/media/emre/Backup/AudioLLM/Datasets")
).resolve()

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset_manifest_common import load_key_to_audio, read_jsonl, write_jsonl
from prepare_daic_woz import (
    DEFAULT_PROMPT_AUDIO_ONLY,
    DEFAULT_REPLY_TEXT,
    build_prompt,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the upstream SECap model on DAIC-WOZ split audio, save "
            "emotion-caption metadata, and optionally rebuild prompt files."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "daic_woz",
        help="DAIC-WOZ manifest root inside Qwen2-Audio-finetune.",
    )
    parser.add_argument(
        "--secap-root",
        type=Path,
        default=None,
        help=(
            "Path to a local SECap checkout containing model.ckpt and weights/. "
            "If omitted, uses $SECAP_ROOT or common workspace locations."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="DAIC-WOZ splits to process.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for SECap. Upstream inference is effectively CUDA-only.",
    )
    parser.add_argument(
        "--metadata-file-name",
        default="secap_metadata.jsonl",
        help="Emotion metadata filename written inside each split directory.",
    )
    parser.add_argument(
        "--transcript-metadata-file-name",
        default="whisper_metadata.jsonl",
        help="Existing split metadata file that provides transcripts.",
    )
    parser.add_argument(
        "--transcript-cache-jsonl",
        type=Path,
        default=Path(os.environ.get("AUDIOLLM_DATASETS_ROOT", "/media/emre/Backup/AudioLLM/Datasets")) / "DAIC-WOZ" / "whisper_transcripts.jsonl",
        help="Fallback transcript cache keyed by absolute audio path.",
    )
    parser.add_argument(
        "--prompt-text",
        default=DEFAULT_PROMPT_AUDIO_ONLY,
        help="Base prompt text. Default auto-upgrades when transcript/emotion is present.",
    )
    parser.add_argument(
        "--reply-text",
        default=DEFAULT_REPLY_TEXT,
        help="Reply instruction appended to prompts.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N samples per split for smoke tests. 0 = all.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate emotion captions even when metadata rows already exist.",
    )
    parser.add_argument(
        "--rebuild-prompts",
        action="store_true",
        help="Rewrite daic_multiprompt.jsonl and daic_woz_multiprompt.jsonl from generated metadata.",
    )
    return parser.parse_args()


def resolve_secap_root(secap_root: Path | None) -> Path:
    candidates: list[Path] = []
    if secap_root is not None:
        candidates.append(secap_root)
    env_root = os.environ.get("SECAP_ROOT", "").strip()
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend(
        [
            WORKSPACE_ROOT / "SECap",
            PROJECT_ROOT / "third_party" / "SECap",
            WORKSPACE_ROOT / "third_party" / "SECap",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    checked = ", ".join(str(p) for p in candidates if str(p))
    raise FileNotFoundError(
        "SECap checkout not found. Provide --secap-root or set SECAP_ROOT. "
        f"Checked: {checked}"
    )


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_transcript_cache(cache_path: Path) -> dict[str, str]:
    if not cache_path.exists():
        return {}
    mapping: dict[str, str] = {}
    for row in read_jsonl(cache_path):
        audio_path = str(row.get("audio_path", "")).strip()
        transcript = str(row.get("transcript", "")).strip()
        if audio_path and transcript:
            p = Path(audio_path)
            if not p.is_absolute():
                p = (DATASETS_ROOT / p).resolve()
            mapping[str(p.resolve())] = transcript
    return mapping


def load_split_transcripts(metadata_path: Path) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    if not metadata_path.exists():
        return mapping
    for row in read_jsonl(metadata_path):
        key = str(row.get("key", "")).strip()
        if not key:
            continue
        mapping[key] = {
            "task": str(row.get("task", "")).strip(),
            "transcript": str(row.get("transcript", "")).strip(),
            "audio_path": str(row.get("audio_path", "")).strip(),
        }
    return mapping


def load_existing_metadata(metadata_path: Path) -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    if not metadata_path.exists():
        return mapping
    for row in read_jsonl(metadata_path):
        key = str(row.get("key", "")).strip()
        if key:
            mapping[key] = row
    return mapping


def preferred_file(split_dir: Path, preferred_name: str, fallback_name: str) -> Path:
    preferred = split_dir / preferred_name
    if preferred.exists():
        return preferred
    return split_dir / fallback_name


def load_split_records(split_dir: Path) -> list[dict]:
    multitask_path = preferred_file(
        split_dir, "daic_woz_multitask.jsonl"
    )
    scp_path = preferred_file(split_dir, "daic_woz.scp")
    key_to_audio = load_key_to_audio(scp_path, PROJECT_ROOT)
    records: list[dict] = []
    for row in read_jsonl(multitask_path):
        key = str(row.get("key", "")).strip()
        task = str(row.get("task", "")).strip()
        audio_path = key_to_audio.get(key, "")
        if key and task and audio_path:
            records.append({"key": key, "task": task, "audio_path": audio_path})
    return records


def load_waveform(audio_path: Path) -> np.ndarray:
    waveform, sample_rate = sf.read(str(audio_path))
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sample_rate != 16000:
        waveform = (
            torchaudio.transforms.Resample(sample_rate, 16000)(
                torch.from_numpy(waveform).unsqueeze(0)
            )
            .squeeze(0)
            .numpy()
        )
    return waveform.astype(np.float32, copy=False)


def normalize_caption(text: str) -> str:
    cleaned = " ".join(str(text).strip().split())
    return cleaned.strip("[]'\"")


def relpath_to_datasets(path: Path) -> str:
    return Path(os.path.relpath(path.resolve(), DATASETS_ROOT)).as_posix()


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


class SecapRunner:
    def __init__(self, secap_root: Path, device: str):
        self.secap_root = secap_root.resolve()
        self.device = device
        if self.device.startswith("cpu"):
            raise RuntimeError(
                "SECap upstream inference casts tensors to fp16 inside generation and "
                "is not reliable on CPU. Use --device cuda or --device cuda:0."
            )
        required = [
            self.secap_root / "model2.py",
            self.secap_root / "model.ckpt",
            self.secap_root / "weights",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "SECap assets missing. Expected model2.py, model.ckpt, and weights/. "
                f"Missing: {', '.join(missing)}"
            )
        if str(self.secap_root) not in sys.path:
            sys.path.insert(0, str(self.secap_root))
        model2 = importlib.import_module("model2")
        motion_audio_cls = getattr(model2, "MotionAudio")
        self.model = motion_audio_cls()
        state_dict = torch.load(
            self.secap_root / "model.ckpt",
            map_location=torch.device("cpu"),
        )
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(torch.device(self.device))
        self.model.eval()

    def infer(self, audio_path: Path) -> tuple[str, list[str], str]:
        waveform = load_waveform(audio_path)
        with torch.no_grad():
            candidates, prompt = self.model.inference([waveform])
        captions = dedupe_keep_order([normalize_caption(x) for x in candidates])
        emotion = captions[0] if captions else ""
        return emotion, captions, str(prompt).strip()


def build_prompt_rows(
    records: list[dict],
    metadata_by_key: dict[str, dict],
    prompt_text: str,
    reply_text: str,
) -> list[dict]:
    rows: list[dict] = []
    for record in records:
        metadata = metadata_by_key.get(record["key"], {})
        rows.append(
            {
                "task": record["task"],
                "prompt": build_prompt(
                    prompt_text,
                    transcript=str(metadata.get("transcript", "")).strip(),
                    emotion=str(metadata.get("emotion", "")).strip(),
                    reply_text=reply_text,
                ),
            }
        )
    return rows


def process_split(
    split_dir: Path,
    runner: SecapRunner,
    transcript_cache: dict[str, str],
    metadata_file_name: str,
    transcript_metadata_file_name: str,
    prompt_text: str,
    reply_text: str,
    overwrite: bool,
    rebuild_prompts: bool,
    limit: int,
):
    records = load_split_records(split_dir)
    if limit > 0:
        records = records[:limit]

    transcript_meta = load_split_transcripts(split_dir / transcript_metadata_file_name)
    metadata_path = split_dir / metadata_file_name
    existing_by_key = load_existing_metadata(metadata_path)
    out_rows: list[dict] = []
    generated = 0
    reused = 0

    for record in records:
        key = record["key"]
        audio_path = Path(record["audio_path"]).resolve()
        transcript_row = transcript_meta.get(key, {})
        transcript = str(transcript_row.get("transcript", "")).strip()
        if not transcript:
            transcript = transcript_cache.get(str(audio_path), "")

        existing = existing_by_key.get(key)
        if existing and str(existing.get("emotion", "")).strip() and not overwrite:
            row = dict(existing)
            row["task"] = record["task"]
            row["audio_path"] = relpath_to_datasets(audio_path)
            if transcript and not str(row.get("transcript", "")).strip():
                row["transcript"] = transcript
            out_rows.append(row)
            reused += 1
            continue

        emotion, candidates, secap_prompt = runner.infer(audio_path)
        out_rows.append(
            {
                "key": key,
                "task": record["task"],
                "dataset": "daic_woz",
                "audio_path": relpath_to_datasets(audio_path),
                "transcript": transcript,
                "emotion": emotion,
                "emotion_candidates": candidates,
                "secap_prompt": secap_prompt,
                "emotion_source": "SECap",
            }
        )
        generated += 1
        print(f"[{split_dir.name}] {key}: {emotion}")

    write_jsonl(metadata_path, out_rows)
    print(f"[{split_dir.name}] wrote metadata={metadata_path} generated={generated} reused={reused}")

    if rebuild_prompts:
        metadata_by_key = {str(row["key"]).strip(): row for row in out_rows}
        prompt_rows = build_prompt_rows(records, metadata_by_key, prompt_text, reply_text)
        prompt_path = split_dir / "daic_woz_multiprompt.jsonl"
        write_jsonl(prompt_path, prompt_rows)
        print(f"[{split_dir.name}] wrote prompts={prompt_path}")



def main():
    args = parse_args()
    secap_root = resolve_secap_root(args.secap_root)
    device = resolve_device(args.device)
    transcript_cache = load_transcript_cache(args.transcript_cache_jsonl)
    runner = SecapRunner(secap_root, device)

    for split in args.splits:
        split_dir = args.dataset_root / split
        if not split_dir.exists():
            print(f"[WARN] skip missing split dir: {split_dir}")
            continue
        process_split(
            split_dir=split_dir,
            runner=runner,
            transcript_cache=transcript_cache,
            metadata_file_name=args.metadata_file_name,
            transcript_metadata_file_name=args.transcript_metadata_file_name,
            prompt_text=args.prompt_text,
            reply_text=args.reply_text,
            overwrite=args.overwrite,
            rebuild_prompts=args.rebuild_prompts,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
