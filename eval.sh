#!/bin/bash
# ============================================================
# Per-Dataset Evaluation Script (MN5 cluster)
# ============================================================
# Evaluates a saved model on the requested held-out test split by default,
# reporting separate metrics for DAIC-WOZ, EATD, and CMDC.
#
# Usage:
#   bash eval.sh                     # uses defaults below
#   bash eval.sh /path/to/best       # evaluate a LoRA checkpoint
# Env switches:
#   MODEL_FAMILY=audio|text
#   PROMPT_MODE=full|audiotext|textonly
#   CHECKPOINT_MODE=auto|full_audio
#   ADAPTER_PATH=/path/to/audio_adapter_state.pt
#   DATASET_NAME=merged|daic_woz|eatd|cmdc
#   DATA_SPLIT=test
#   ALLOW_NON_TEST_EVAL=1
#   SKIP_SPLIT_VALIDATION=1
#   CMDC_FOLD=fold1             # legacy only; clean holdout uses data/cmdc/$DATA_SPLIT
#   DAIC_WOZ_EVAL_LEVEL=person|segment
#   EATD_EVAL_LEVEL=person|segment
#   CMDC_EVAL_LEVEL=person|segment
#   LOG_DIR=/custom/log/dir
#   RESULTS_DIR=/custom/results/dir
#   OUTPUT_JSON=/custom/results.json
#   EVAL_NAME=custom_eval_name
# ============================================================

set -e
set -o pipefail

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$LOCAL_DIR" || exit 1

PEFT_PATH="${1:-${PEFT_PATH:-$LOCAL_DIR/output_model/1e-05_20260330_023140/07-15}}"
MODEL_FAMILY="${MODEL_FAMILY:-audio}"
CHECKPOINT_MODE="${CHECKPOINT_MODE:-auto}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
DATASET_NAME="${DATASET_NAME:-merged}"
DATA_SPLIT="${DATA_SPLIT:-test}"
CMDC_FOLD="${CMDC_FOLD:-fold1}"
DAIC_WOZ_EVAL_LEVEL="${DAIC_WOZ_EVAL_LEVEL:-${DAIC_EVAL_LEVEL:-person}}"
DAIC_WOZ_EVAL_MODE="${DAIC_WOZ_EVAL_MODE:-${DAIC_EVAL_MODE:-majority_vote}}"
DAIC_WOZ_PERSON_THRESHOLD="${DAIC_WOZ_PERSON_THRESHOLD:-${DAIC_PERSON_THRESHOLD:-0.5}}"
EATD_EVAL_LEVEL="${EATD_EVAL_LEVEL:-person}"
EATD_EVAL_MODE="${EATD_EVAL_MODE:-majority_vote}"
EATD_PERSON_THRESHOLD="${EATD_PERSON_THRESHOLD:-0.5}"
CMDC_EVAL_LEVEL="${CMDC_EVAL_LEVEL:-person}"
CMDC_EVAL_MODE="${CMDC_EVAL_MODE:-majority_vote}"
CMDC_PERSON_THRESHOLD="${CMDC_PERSON_THRESHOLD:-0.5}"

BATCH_SIZE="${BATCH_SIZE:-1}"
DEVICE="${DEVICE:-cuda:0}"

case "${DATA_SPLIT}" in
    train|val|test)
        ;;
    *)
        echo "Unsupported DATA_SPLIT: $DATA_SPLIT"
        echo "Use DATA_SPLIT=train, DATA_SPLIT=val, or DATA_SPLIT=test"
        exit 1
        ;;
esac

if [[ "$DATA_SPLIT" != "test" ]]; then
    case "${ALLOW_NON_TEST_EVAL:-0}" in
        1|true|TRUE|yes|YES|on|ON)
            echo "[Warning] Running evaluation on DATA_SPLIT=$DATA_SPLIT because ALLOW_NON_TEST_EVAL is enabled."
            ;;
        *)
            echo "Refusing final evaluation on DATA_SPLIT=$DATA_SPLIT."
            echo "Final reporting must use DATA_SPLIT=test. Set ALLOW_NON_TEST_EVAL=1 only for debugging."
            exit 1
            ;;
    esac
fi

normalize_model_family() {
    local raw="$1"
    case "$raw" in
        textonly) echo "text" ;;
        *) echo "$raw" ;;
    esac
}

sanitize_name() {
    local value="$1"
    value="${value// /_}"
    value="${value//\//_}"
    value="${value//:/_}"
    value="${value//[^A-Za-z0-9._-]/_}"
    echo "$value"
}

derive_checkpoint_name() {
    local checkpoint_path="$1"
    local normalized="${checkpoint_path%/}"
    local base_name
    base_name="$(basename "$normalized")"

    if [[ "$base_name" == "best_model" ]]; then
        local parent_name
        parent_name="$(basename "$(dirname "$normalized")")"
        if [[ -n "$parent_name" ]]; then
            echo "$parent_name"
            return
        fi
    fi

    if [[ -n "$base_name" ]]; then
        echo "$base_name"
        return
    fi

    echo "base_model"
}

MODEL_FAMILY="$(normalize_model_family "$MODEL_FAMILY")"

case "$MODEL_FAMILY" in
    audio)
        export MODEL_PATH="${MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen2-Audio-7B-Instruct}"
        EVAL_SCRIPT="evaluate_per_dataset.py"
        PROMPT_MODE="${PROMPT_MODE:-audiotext}"
        ;;
    text)
        export MODEL_PATH="${MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen2-7B-Instruct}"
        EVAL_SCRIPT="evaluate_textonly.py"
        PROMPT_MODE="${PROMPT_MODE:-textonly}"
        ;;
    *)
        echo "Unsupported MODEL_FAMILY: $MODEL_FAMILY"
        echo "Use MODEL_FAMILY=audio or MODEL_FAMILY=text"
        exit 1
        ;;
esac

case "${MODEL_FAMILY}:${PROMPT_MODE}" in
    audio:full|audio:audiotext|text:textonly)
        ;;
    *)
        echo "Invalid MODEL_FAMILY / PROMPT_MODE combination: ${MODEL_FAMILY} + ${PROMPT_MODE}"
        echo "Allowed combinations are: audio+full, audio+audiotext, text+textonly"
        exit 1
        ;;
esac

case "$DATASET_NAME" in
    merged|daic_woz|eatd|cmdc)
        ;;
    *)
        echo "Unsupported DATASET_NAME: $DATASET_NAME"
        echo "Use DATASET_NAME=merged, DATASET_NAME=daic_woz, DATASET_NAME=eatd, or DATASET_NAME=cmdc"
        exit 1
        ;;
esac

case "$PROMPT_MODE" in
    full|merged|default)
        PROMPT_FILE_SUFFIX="multiprompt.jsonl"
        ;;
    audiotext)
        PROMPT_FILE_SUFFIX="multiprompt_audiotext.jsonl"
        ;;
    textonly)
        PROMPT_FILE_SUFFIX="multiprompt_textonly.jsonl"
        ;;
    *)
        echo "Unsupported PROMPT_MODE: $PROMPT_MODE"
        echo "Use PROMPT_MODE=full, PROMPT_MODE=audiotext, or PROMPT_MODE=textonly"
        exit 1
        ;;
esac

case "$DATASET_NAME" in
    merged)
        DATA_BASENAME="merged"
        DATA_SUBDIR="merged/$DATA_SPLIT"
        ;;
    daic_woz)
        DATA_BASENAME="daic_woz"
        DATA_SUBDIR="daic_woz/$DATA_SPLIT"
        ;;
    eatd)
        DATA_BASENAME="eatd"
        DATA_SUBDIR="eatd/$DATA_SPLIT"
        ;;
    cmdc)
        DATA_BASENAME="cmdc"
        DATA_SUBDIR="cmdc/$DATA_SPLIT"
        ;;
esac

DATA_PATH="${DATA_PATH:-$LOCAL_DIR/data/$DATA_SUBDIR}"
SCP_FILENAME="${SCP_FILENAME:-$DATA_BASENAME.scp}"
TASK_FILENAME="${TASK_FILENAME:-${DATA_BASENAME}_multitask.jsonl}"
PROMPT_FILE_DEFAULT="${DATA_BASENAME}_${PROMPT_FILE_SUFFIX}"
PROMPT_PATH="${PROMPT_PATH:-$DATA_PATH/$PROMPT_FILE_DEFAULT}"

DAIC_LEVEL_SUFFIX=""
if [[ "$DATASET_NAME" == "daic_woz" ]]; then
    DAIC_LEVEL_SUFFIX="_${DAIC_WOZ_EVAL_LEVEL}"
fi

CHECKPOINT_NAME="$(sanitize_name "$(derive_checkpoint_name "$PEFT_PATH")")"
if [[ -z "$CHECKPOINT_NAME" ]]; then
    CHECKPOINT_NAME="checkpoint"
fi

DEFAULT_EVAL_BASENAME="${DATASET_NAME}${DAIC_LEVEL_SUFFIX}_${MODEL_FAMILY}_${PROMPT_MODE}_${CHECKPOINT_NAME}"
EVAL_BASENAME_RAW="${EVAL_NAME:-$DEFAULT_EVAL_BASENAME}"
EVAL_BASENAME="$(sanitize_name "$EVAL_BASENAME_RAW")"
if [[ -z "$EVAL_BASENAME" ]]; then
    EVAL_BASENAME="eval_run"
fi

EVAL_TIMESTAMP="${EVAL_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_TIMESTAMP="${LOG_TIMESTAMP:-$(date +%Y-%m-%d_%H:%M:%S)}"
EVAL_NAME_RESOLVED="${EVAL_BASENAME}_${EVAL_TIMESTAMP}"

LOG_DIR="${LOG_DIR:-$LOCAL_DIR/logs/eval_${DATASET_NAME}${DAIC_LEVEL_SUFFIX}}"
RESULTS_DIR="${RESULTS_DIR:-$LOCAL_DIR/eval_results/eval_${DATASET_NAME}${DAIC_LEVEL_SUFFIX}}"
OUTPUT_JSON="${OUTPUT_JSON:-$RESULTS_DIR/${EVAL_NAME_RESOLVED}_results.json}"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

if [[ "${SKIP_SPLIT_VALIDATION:-0}" != "1" ]]; then
    python tools/validate_splits.py --dataset "$DATASET_NAME" --strict
fi

echo "============================================"
echo "  Per-Dataset Evaluation"
echo "============================================"
echo "  DATASET_NAME    : $DATASET_NAME"
echo "  DATA_SPLIT      : $DATA_SPLIT"
echo "  CMDC_FOLD       : $CMDC_FOLD"
echo "  MODEL_FAMILY    : $MODEL_FAMILY"
echo "  PROMPT_MODE     : $PROMPT_MODE"
echo "  DAIC_LEVEL      : $DAIC_WOZ_EVAL_LEVEL"
echo "  DAIC_MODE       : $DAIC_WOZ_EVAL_MODE"
echo "  DAIC_THRESHOLD  : $DAIC_WOZ_PERSON_THRESHOLD"
echo "  EATD_LEVEL      : $EATD_EVAL_LEVEL"
echo "  EATD_MODE       : $EATD_EVAL_MODE"
echo "  EATD_THRESHOLD  : $EATD_PERSON_THRESHOLD"
echo "  CMDC_LEVEL      : $CMDC_EVAL_LEVEL"
echo "  CMDC_MODE       : $CMDC_EVAL_MODE"
echo "  CMDC_THRESHOLD  : $CMDC_PERSON_THRESHOLD"
echo "  CHECKPOINT_MODE : $CHECKPOINT_MODE"
echo "  EVAL_SCRIPT     : $EVAL_SCRIPT"
echo "  MODEL_PATH      : $MODEL_PATH"
echo "  PEFT_PATH       : $PEFT_PATH"
echo "  CHECKPOINT_NAME : $CHECKPOINT_NAME"
echo "  EVAL_NAME       : $EVAL_NAME_RESOLVED"
echo "  LOG_DIR         : $LOG_DIR"
echo "  RESULTS_DIR     : $RESULTS_DIR"
echo "  OUTPUT_JSON     : $OUTPUT_JSON"
if [[ -n "$ADAPTER_PATH" ]]; then
    echo "  ADAPTER_PATH    : $ADAPTER_PATH"
else
    echo "  ADAPTER_PATH    : (auto)"
fi
echo "  DATA_PATH       : $DATA_PATH"
echo "  PROMPT_PATH     : $PROMPT_PATH"
echo "  DEVICE          : $DEVICE"
echo "============================================"

CMD=(
    python "$EVAL_SCRIPT"
    --model_path "$MODEL_PATH"
    --dataset_name "$DATASET_NAME"
    --daic_woz_eval_level "$DAIC_WOZ_EVAL_LEVEL"
    --daic_woz_eval_mode "$DAIC_WOZ_EVAL_MODE"
    --daic_woz_person_threshold "$DAIC_WOZ_PERSON_THRESHOLD"
    --eatd_eval_level "$EATD_EVAL_LEVEL"
    --eatd_eval_mode "$EATD_EVAL_MODE"
    --eatd_person_threshold "$EATD_PERSON_THRESHOLD"
    --cmdc_eval_level "$CMDC_EVAL_LEVEL"
    --cmdc_eval_mode "$CMDC_EVAL_MODE"
    --cmdc_person_threshold "$CMDC_PERSON_THRESHOLD"
    --data_path "$DATA_PATH"
    --prompt_path "$PROMPT_PATH"
    --data_split "$DATA_SPLIT"
    --batch_size "$BATCH_SIZE"
    --device "$DEVICE"
    --output_json "$OUTPUT_JSON"
)

if [[ "$MODEL_FAMILY" == "audio" ]]; then
    CMD+=(
        --scp_filename "$SCP_FILENAME"
        --task_filename "$TASK_FILENAME"
        --checkpoint_mode "$CHECKPOINT_MODE"
        --peft_path "$PEFT_PATH"
    )
    if [[ -n "$ADAPTER_PATH" ]]; then
        CMD+=(--adapter_path "$ADAPTER_PATH")
    fi
else
    CMD+=(--task_filename "$TASK_FILENAME")
    if [[ -n "$PEFT_PATH" && "$PEFT_PATH" != "none" && "$PEFT_PATH" != "null" && "$PEFT_PATH" != "base" && "$PEFT_PATH" != "baseline" ]]; then
        CMD+=(--peft_path "$PEFT_PATH")
    fi
fi

"${CMD[@]}"
