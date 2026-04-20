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
# Env switches:
#   MODEL_FAMILY=audio|text
#   PROMPT_MODE=full|audiotext|textonly
# ============================================================

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$LOCAL_DIR" || exit 1

# --- Configurable paths (match train.sh conventions) ---
PEFT_PATH="${1:-$LOCAL_DIR/output_model/1e-05_20260330_023140/07-15}"
MODEL_FAMILY="${MODEL_FAMILY:-audio}"
PROMPT_MODE="${PROMPT_MODE:-full}"

DATA_PATH="${DATA_PATH:-$LOCAL_DIR/data/merged/val}"
SCP_FILENAME="${SCP_FILENAME:-merged.scp}"
TASK_FILENAME="${TASK_FILENAME:-merged_multitask.jsonl}"

BATCH_SIZE="${BATCH_SIZE:-1}"
DEVICE="${DEVICE:-cuda:0}"

case "$MODEL_FAMILY" in
    audio)
        export MODEL_PATH="${MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen2-Audio-7B-Instruct}"
        EVAL_SCRIPT="evaluate_per_dataset.py"
        ;;
    text|textonly)
        export MODEL_PATH="${MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen2-7B-Instruct}"
        EVAL_SCRIPT="evaluate_textonly.py"
        ;;
    *)
        echo "Unsupported MODEL_FAMILY: $MODEL_FAMILY"
        echo "Use MODEL_FAMILY=audio or MODEL_FAMILY=text"
        exit 1
        ;;
esac

case "$PROMPT_MODE" in
    full|merged|default)
        PROMPT_FILE_DEFAULT="merged_multiprompt.jsonl"
        ;;
    audiotext)
        PROMPT_FILE_DEFAULT="merged_multiprompt_audiotext.jsonl"
        ;;
    textonly)
        PROMPT_FILE_DEFAULT="merged_multiprompt_textonly.jsonl"
        ;;
    *)
        echo "Unsupported PROMPT_MODE: $PROMPT_MODE"
        echo "Use PROMPT_MODE=full, PROMPT_MODE=audiotext, or PROMPT_MODE=textonly"
        exit 1
        ;;
esac

PROMPT_PATH="${PROMPT_PATH:-$DATA_PATH/$PROMPT_FILE_DEFAULT}"

echo "============================================"
echo "  Per-Dataset Evaluation"
echo "============================================"
echo "  MODEL_FAMILY : $MODEL_FAMILY"
echo "  PROMPT_MODE  : $PROMPT_MODE"
echo "  EVAL_SCRIPT  : $EVAL_SCRIPT"
echo "  MODEL_PATH : $MODEL_PATH"
if [[ -z "$PEFT_PATH" || "$PEFT_PATH" == "none" || "$PEFT_PATH" == "null" || "$PEFT_PATH" == "base" || "$PEFT_PATH" == "baseline" ]]; then
    echo "  PEFT_PATH  : (none - base model)"
else
    echo "  PEFT_PATH  : $PEFT_PATH"
fi
echo "  DATA_PATH  : $DATA_PATH"
echo "  PROMPT_PATH: $PROMPT_PATH"
echo "  DEVICE     : $DEVICE"
echo "============================================"

# --- Run evaluation ---
CMD=(
    python "$EVAL_SCRIPT"
    --model_path "$MODEL_PATH"
    --data_path "$DATA_PATH"
    --prompt_path "$PROMPT_PATH"
    --batch_size "$BATCH_SIZE"
    --device "$DEVICE"
)

if [[ "$MODEL_FAMILY" == "audio" ]]; then
    CMD+=(
        --scp_filename "$SCP_FILENAME"
        --task_filename "$TASK_FILENAME"
    )
else
    CMD+=(--task_filename "$TASK_FILENAME")
fi

if [[ -n "$PEFT_PATH" && "$PEFT_PATH" != "none" && "$PEFT_PATH" != "null" && "$PEFT_PATH" != "base" && "$PEFT_PATH" != "baseline" ]]; then
    CMD+=(--peft_path "$PEFT_PATH")
fi

"${CMD[@]}"
