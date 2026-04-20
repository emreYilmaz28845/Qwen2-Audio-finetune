#!/bin/bash
# ============================================================
# Per-Dataset Evaluation Script (MN5 cluster)
# ============================================================
# Evaluates the best saved model on the merged validation set,
# reporting separate metrics for DAIC-WOZ, EATD, and CMDC.
#
# Usage:
#   bash eval.sh                     # uses defaults below
#   bash eval.sh /path/to/best       # evaluate a LoRA checkpoint
#   bash eval.sh none                # evaluate the base model without LoRA
# ============================================================

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$LOCAL_DIR" || exit 1

# --- Configurable paths (match train.sh conventions) ---
PEFT_PATH="${1:-$LOCAL_DIR/output_model/1e-05_20260330_023140/07-15}"

MODEL_PATH="${MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen2-Audio-7B-Instruct}"
DATA_PATH="${DATA_PATH:-$LOCAL_DIR/data/merged/val}"
PROMPT_PATH="${PROMPT_PATH:-$LOCAL_DIR/data/merged/val/merged_multiprompt.jsonl}"
SCP_FILENAME="${SCP_FILENAME:-merged.scp}"
TASK_FILENAME="${TASK_FILENAME:-merged_multitask.jsonl}"

BATCH_SIZE="${BATCH_SIZE:-1}"
DEVICE="${DEVICE:-cuda:0}"

echo "============================================"
echo "  Per-Dataset Evaluation (TEXT-ONLY mode)"
echo "============================================"
echo "  MODEL_PATH : $MODEL_PATH"
if [[ -z "$PEFT_PATH" || "$PEFT_PATH" == "none" || "$PEFT_PATH" == "null" || "$PEFT_PATH" == "base" || "$PEFT_PATH" == "baseline" ]]; then
    echo "  PEFT_PATH  : (none - base model)"
else
    echo "  PEFT_PATH  : $PEFT_PATH"
fi
echo "  DATA_PATH  : $DATA_PATH"
echo "  DEVICE     : $DEVICE"
echo "============================================"

# --- Run evaluation ---
CMD=(
    python evaluate_per_dataset.py
    --model_path "$MODEL_PATH"
    --data_path "$DATA_PATH"
    --prompt_path "$PROMPT_PATH"
    --scp_filename "$SCP_FILENAME"
    --task_filename "$TASK_FILENAME"
    --batch_size "$BATCH_SIZE"
    --device "$DEVICE"
)

if [[ -n "$PEFT_PATH" && "$PEFT_PATH" != "none" && "$PEFT_PATH" != "null" && "$PEFT_PATH" != "base" && "$PEFT_PATH" != "baseline" ]]; then
    CMD+=(--peft_path "$PEFT_PATH")
fi

"${CMD[@]}"
