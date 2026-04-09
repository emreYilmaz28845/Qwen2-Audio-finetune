#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, pstdev


EVAL_METRICS_RE = re.compile(
    r"Loss:\s*([-+]?\d*\.?\d+),\s*"
    r"Acc:\s*([-+]?\d*\.?\d+),\s*"
    r"Prec:\s*([-+]?\d*\.?\d+),\s*"
    r"Rec:\s*([-+]?\d*\.?\d+),\s*"
    r"F1:\s*([-+]?\d*\.?\d+),\s*"
    r"wF1:\s*([-+]?\d*\.?\d+)"
)
SAVE_RE = re.compile(
    r"\[Saving\]\s+Better F1\s+([-+]?\d*\.?\d+)\s+>\s+[-+a-zA-Z0-9\.]+:\s+(.+)"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, help="Root directory containing fold1..fold5")
    parser.add_argument("--folds", nargs="+", required=True, help="Fold names, e.g. fold1 fold2")
    parser.add_argument("--json-out", required=True, help="Output JSON path")
    parser.add_argument("--csv-out", required=True, help="Output CSV path")
    return parser.parse_args()


def find_latest_train_log(fold_dir: Path):
    logs = sorted(
        fold_dir.rglob("train_log"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return logs[0] if logs else None


def parse_fold_log(log_path: Path, fold_name: str):
    latest_eval = None
    best = None

    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            metrics_match = EVAL_METRICS_RE.search(line)
            if metrics_match:
                latest_eval = {
                    "eval_loss": float(metrics_match.group(1)),
                    "eval_acc": float(metrics_match.group(2)),
                    "eval_precision": float(metrics_match.group(3)),
                    "eval_recall": float(metrics_match.group(4)),
                    "eval_f1": float(metrics_match.group(5)),
                    "eval_weighted_f1": float(metrics_match.group(6)),
                }
                continue

            save_match = SAVE_RE.search(line)
            if save_match and latest_eval is not None:
                best = {
                    "fold": fold_name,
                    "log_path": str(log_path),
                    "best_model_path": save_match.group(2).strip(),
                    **latest_eval,
                }

    if best is None:
        return {
            "fold": fold_name,
            "log_path": str(log_path),
            "best_model_path": None,
            "eval_loss": None,
            "eval_acc": None,
            "eval_precision": None,
            "eval_recall": None,
            "eval_f1": None,
            "eval_weighted_f1": None,
        }

    return best


def safe_stats(values):
    if not values:
        return {"mean": None, "std": None}
    if len(values) == 1:
        return {"mean": mean(values), "std": 0.0}
    return {"mean": mean(values), "std": pstdev(values)}


def build_summary(fold_results):
    metric_map = {
        "eval_loss": "loss",
        "eval_acc": "acc",
        "eval_precision": "precision",
        "eval_recall": "recall",
        "eval_f1": "f1",
        "eval_weighted_f1": "weighted_f1",
    }

    summary = {
        "num_folds_requested": len(fold_results),
        "num_folds_completed": sum(1 for row in fold_results if row["eval_f1"] is not None),
    }

    for field, label in metric_map.items():
        values = [row[field] for row in fold_results if row[field] is not None]
        stats = safe_stats(values)
        summary[f"{label}_mean"] = stats["mean"]
        summary[f"{label}_std"] = stats["std"]

    return summary


def write_csv(csv_path: Path, fold_results, summary):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "fold",
        "log_path",
        "best_model_path",
        "eval_loss",
        "eval_acc",
        "eval_precision",
        "eval_recall",
        "eval_f1",
        "eval_weighted_f1",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in fold_results:
            writer.writerow(row)

        writer.writerow({})
        writer.writerow({"fold": "summary", "log_path": f"completed={summary['num_folds_completed']}"})
        writer.writerow({"fold": "loss_mean", "eval_loss": summary["loss_mean"]})
        writer.writerow({"fold": "loss_std", "eval_loss": summary["loss_std"]})
        writer.writerow({"fold": "acc_mean", "eval_acc": summary["acc_mean"]})
        writer.writerow({"fold": "acc_std", "eval_acc": summary["acc_std"]})
        writer.writerow({"fold": "precision_mean", "eval_precision": summary["precision_mean"]})
        writer.writerow({"fold": "precision_std", "eval_precision": summary["precision_std"]})
        writer.writerow({"fold": "recall_mean", "eval_recall": summary["recall_mean"]})
        writer.writerow({"fold": "recall_std", "eval_recall": summary["recall_std"]})
        writer.writerow({"fold": "f1_mean", "eval_f1": summary["f1_mean"]})
        writer.writerow({"fold": "f1_std", "eval_f1": summary["f1_std"]})
        writer.writerow({"fold": "weighted_f1_mean", "eval_weighted_f1": summary["weighted_f1_mean"]})
        writer.writerow({"fold": "weighted_f1_std", "eval_weighted_f1": summary["weighted_f1_std"]})


def main():
    args = parse_args()

    results_root = Path(args.results_root)
    fold_results = []

    for fold_name in args.folds:
        fold_dir = results_root / fold_name
        log_path = find_latest_train_log(fold_dir)
        if log_path is None:
            fold_results.append(
                {
                    "fold": fold_name,
                    "log_path": None,
                    "best_model_path": None,
                    "eval_loss": None,
                    "eval_acc": None,
                    "eval_precision": None,
                    "eval_recall": None,
                    "eval_f1": None,
                    "eval_weighted_f1": None,
                }
            )
            continue

        fold_results.append(parse_fold_log(log_path, fold_name))

    summary = build_summary(fold_results)

    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "results_root": str(results_root),
                "fold_results": fold_results,
                "summary": summary,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    write_csv(Path(args.csv_out), fold_results, summary)

    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
