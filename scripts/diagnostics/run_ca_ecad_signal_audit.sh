#!/usr/bin/env bash
set -euo pipefail

# Frozen Step-N signal audit.  This only generates grouped trajectories and
# computes CA-ECAD credits offline; it never computes log-probs or updates an
# actor, critic, optimizer, scheduler, or checkpoint.

: "${STORAGE_ROOT:?Set STORAGE_ROOT to the data-disk root.}"
: "${BASE_MODEL:?Set BASE_MODEL to the frozen actor checkpoint to audit.}"
: "${PRIOR_STATE_PATH:?Set PRIOR_STATE_PATH to ca_ecad/global_step_N.json.}"

DIAGNOSTICS_OUTPUT_DIR="${DIAGNOSTICS_OUTPUT_DIR:-${STORAGE_ROOT}/results/ca_ecad_signal_audit}"
CALIBRATION_PROMPTS="${CALIBRATION_PROMPTS:-256}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
N_AGENT="${N_AGENT:-5}"

if [[ -e "${DIAGNOSTICS_OUTPUT_DIR}/ca_ecad_signal_audit.json" ]]; then
  echo "Refusing to overwrite existing signal audit: ${DIAGNOSTICS_OUTPUT_DIR}" >&2
  exit 2
fi

STORAGE_ROOT="${STORAGE_ROOT}" \
BASE_MODEL="${BASE_MODEL}" \
NUM_GPUS="${NUM_GPUS:-2}" \
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
N_AGENT="${N_AGENT}" \
CALIBRATION_PROMPTS="${CALIBRATION_PROMPTS}" \
DIAGNOSTICS_OUTPUT_DIR="${DIAGNOSTICS_OUTPUT_DIR}" \
DATA_DIR="${DATA_DIR:-${STORAGE_ROOT}/data/nq_hotpotqa_train}" \
TRAIN_FILE="${TRAIN_FILE:-${STORAGE_ROOT}/data/nq_hotpotqa_train/train.parquet}" \
VAL_FILE="${VAL_FILE:-${STORAGE_ROOT}/data/nq_search/test.parquet}" \
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}" \
WANDB_PROJECT="${WANDB_PROJECT:-CA-ECAD-Search-Agent}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME:-ca-ecad-step100-signal-audit}" \
bash scripts/diagnostics/run_ca_ecad_phase2.sh

python3 scripts/diagnostics/analyze_ca_ecad_signal.py \
  --input-dir "${DIAGNOSTICS_OUTPUT_DIR}" \
  --prior-state-path "${PRIOR_STATE_PATH}" \
  --output-dir "${DIAGNOSTICS_OUTPUT_DIR}" \
  --alpha 2.0 \
  --eta 0.25 \
  --kappa 1.0
