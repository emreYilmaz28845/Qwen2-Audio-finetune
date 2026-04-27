#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/emre/Projects/AudioLLM/Qwen2-Audio-finetune"
DATA_ROOT="$PROJECT_ROOT/data/cmdc"
PAPER_ROOT="$PROJECT_ROOT/data/cmdc-paper"
DATASETS_ROOT="/media/emre/Backup/AudioLLM/Datasets"
CMDC_DATASET_ROOT="$DATASETS_ROOT/CMDC"

export AUDIOLLM_DATASETS_ROOT="$DATASETS_ROOT"

python "$PROJECT_ROOT/src/generate_scp_cmdc.py"
python "$PROJECT_ROOT/src/generate_multitask_cmdc.py" \
  --scp_root "$DATA_ROOT"
python "$PROJECT_ROOT/src/generate_multiprompt_cmdc.py" \
  --cmdc_root "$DATA_ROOT" \
  --paper_root "$PAPER_ROOT" \
  --dataset_root "$CMDC_DATASET_ROOT"
python "$PROJECT_ROOT/tools/make_audiotext_multiprompt.py" \
  --generate-cmdc-5fold \
  --cmdc-root "$DATA_ROOT"
python "$PROJECT_ROOT/tools/make_textonly_multiprompt.py" \
  --generate-cmdc-5fold \
  --cmdc-root "$DATA_ROOT"

echo "CMDC manifests regenerated successfully."
