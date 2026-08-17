#!/usr/bin/env bash
set -euo pipefail

# V7 Phase-2 geometry/estimator audit. Each outer update constructs an
# ordinary GRPO proposal, scores the same K=16 cached probes twice, writes the
# per-state diagnostics, then restores theta/AdamW and skips the scheduler.
# It is therefore intentionally not a training command.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export NUM_GPUS="${NUM_GPUS:-2}"
export STORAGE_ROOT="${STORAGE_ROOT:-/mnt/data1/zar/eitr_storage}"

: "${BASE_MODEL:?Set BASE_MODEL to the frozen lightweight checkpoint (for example global_step_800)}"

AUDIT_BATCHES="${AUDIT_BATCHES:-1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-eitr-v7-phase2-geometry-audit-k4-k16-v1}"
AUDIT_OUTPUT_DIR="${AUDIT_OUTPUT_DIR:-$STORAGE_ROOT/evaluations/$EXPERIMENT_NAME}"
EITR_V7_ARTIFACT_PATH="${EITR_V7_ARTIFACT_PATH:-$AUDIT_OUTPUT_DIR/v7_geometry_batches.jsonl}"

if ! [[ "$AUDIT_BATCHES" =~ ^[1-9][0-9]*$ ]]; then
    echo "AUDIT_BATCHES must be a positive integer; got: $AUDIT_BATCHES" >&2
    exit 2
fi
if [[ "${CHECK_ONLY:-false}" != true && -e "$EITR_V7_ARTIFACT_PATH" ]]; then
    echo "Refusing to append to an existing V7 artifact: $EITR_V7_ARTIFACT_PATH" >&2
    exit 2
fi

export EITR_MODE=probe_only
export EITR_V7_GEOMETRY_AUDIT=true
export EITR_PROBE_COUNT=16
export EITR_MIN_VALID_PROBE_COUNT=12
export EITR_V7_PRIMARY_K=4
export EITR_V7_REFERENCE_K=16
export EITR_V7_REFERENCE_MIN_VALID_PROBE_COUNT=12
export EITR_V7_ACCEPT_RADIUS="${EITR_V7_ACCEPT_RADIUS:-0.001}"
export EITR_V7_ARTIFACT_PATH
export EITR_PROBE_MICRO_BATCH_SIZE=16
export EITR_PROBE_LOGPROB_MICRO_BATCH_SIZE=16
export EITR_PROBE_GRADIENT_CHECKPOINTING=false

export TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-$STORAGE_ROOT/data/nq_search}"
export VAL_DATA_DIR="${VAL_DATA_DIR:-$STORAGE_ROOT/data/nq_search}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export TRAIN_DATA_NUM="${TRAIN_DATA_NUM:-$((TRAIN_BATCH_SIZE * AUDIT_BATCHES))}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-4}"
export ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE:-8}"
export REF_LOG_PROB_MICRO_BATCH_SIZE="${REF_LOG_PROB_MICRO_BATCH_SIZE:-8}"
export VAL_DATA_NUM="${VAL_DATA_NUM:-32}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
export N_AGENT="${N_AGENT:-5}"
export SHUFFLE_TRAIN_DATALOADER=false

export ACTOR_LR="${ACTOR_LR:-5e-7}"
export LR_WARMUP_STEPS_RATIO=0
export ACTOR_FSDP_OPTIMIZER_OFFLOAD=true
export ACTOR_BATCH_OFFLOAD=true
export REF_FSDP_PARAM_OFFLOAD=true
export SAVE_FREQ=-1
export TEST_FREQ=-1
export TOTAL_TRAINING_STEPS="$AUDIT_BATCHES"
export TOTAL_EPOCHS="$AUDIT_BATCHES"
export METRICS_LEVEL=debug
export TERMINAL_TRACE_SAMPLES="${TERMINAL_TRACE_SAMPLES:-0}"
export EXPERIMENT_NAME

exec bash scripts/train/train_eitr_nq_gate_c_smoke.sh
