#!/usr/bin/env bash
set -euo pipefail

# Formal Phase-2 Conditional EITR run. The selected initial paper-run ceiling
# is 8,000 outer updates; callers can still override it explicitly.
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-8000}"

export EITR_MODE="${EITR_MODE:-eitr}"
export EITR_PROBE_PROBABILITY="${EITR_PROBE_PROBABILITY:-1.0}"
export EITR_PROBE_COUNT="${EITR_PROBE_COUNT:-4}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-eitr-nq-hotpotqa-v03-phase2-full}"
export METRICS_LEVEL="${METRICS_LEVEL:-core}"
export EITR_POST_DIAGNOSTIC_FREQ="${EITR_POST_DIAGNOSTIC_FREQ:-10}"
FORMAL_STORAGE_ROOT="${STORAGE_ROOT:-/mnt/data1/zar/eitr_storage}"
# Search-R1 trains on the concatenated NQ+HotpotQA pool. Keep paper validation
# on the fixed NQ split so its EM remains directly comparable to prior runs.
export TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-$FORMAL_STORAGE_ROOT/data/nq_hotpotqa_train}"
export VAL_DATA_DIR="${VAL_DATA_DIR:-$FORMAL_STORAGE_ROOT/data/nq_search}"
export SHUFFLE_TRAIN_DATALOADER="${SHUFFLE_TRAIN_DATALOADER:-true}"
export TRAIN_DATA_NUM="${TRAIN_DATA_NUM:-null}"
# Keep periodic validation bounded and reproducible. The trainer samples this
# fixed-size subset with random_state=42 instead of evaluating the full split.
export VAL_DATA_NUM="${VAL_DATA_NUM:-256}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-256}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-16}"
# Cached-old and current probe scores must use the same chunking.  Keep the
# formal default conservative because current scoring now runs in eval mode
# with gradients enabled for the EITR correction.
export EITR_PROBE_LOGPROB_MICRO_BATCH_SIZE="${EITR_PROBE_LOGPROB_MICRO_BATCH_SIZE:-4}"
export EITR_PROBE_MICRO_BATCH_SIZE="${EITR_PROBE_MICRO_BATCH_SIZE:-4}"
# Search-R1 v0.3 warms up for about 286 of its 1,005 outer updates. Preserve
# that update count for the 8,000-step ceiling instead of incorrectly keeping
# the 0.285 ratio (which would stretch warmup to 2,280 very slow EITR steps).
export LR_WARMUP_STEPS_RATIO="${LR_WARMUP_STEPS_RATIO:-0.03575}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
export SAVE_FREQ="${SAVE_FREQ:-100}"
export TEST_FREQ="${TEST_FREQ:-50}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/train_eitr_nq_gate_c_smoke.sh"
