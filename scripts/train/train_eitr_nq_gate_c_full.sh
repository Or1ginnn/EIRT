#!/usr/bin/env bash
set -euo pipefail

# Formal Gate C run. The caller must deliberately choose the training budget;
# this wrapper prevents a smoke-test default from being mistaken for a paper run.
: "${TOTAL_TRAINING_STEPS:?Set TOTAL_TRAINING_STEPS to the Search-R1 baseline budget}"

export EITR_ENABLED="${EITR_ENABLED:-true}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-eitr-nq-gate-c-full}"
export TRAIN_DATA_NUM="${TRAIN_DATA_NUM:-null}"
export VAL_DATA_NUM="${VAL_DATA_NUM:-null}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-256}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-256}"
export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-64}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
export SAVE_FREQ="${SAVE_FREQ:-100}"
export TEST_FREQ="${TEST_FREQ:-100}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/train_eitr_nq_gate_c_smoke.sh"
