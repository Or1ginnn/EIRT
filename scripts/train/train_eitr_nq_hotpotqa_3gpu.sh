#!/usr/bin/env bash
set -euo pipefail

# Safe three-card profile for physical GPUs 1,2,3.  Thirty questions with
# five sibling rollouts gives 150 trajectories: 50 per FSDP rank and five
# global GRPO mini-batches, matching the previous 32x5 / mini-32 semantics.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"
export NUM_GPUS="${NUM_GPUS:-3}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-30}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-30}"
# Keep the faster two-rollout-per-rank GRPO path.  EITR has a separate memory
# peak, so it uses activation checkpointing below instead of slowing every
# ordinary GRPO micro-batch.
export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-6}"
export ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE:-30}"
export REF_LOG_PROB_MICRO_BATCH_SIZE="${REF_LOG_PROB_MICRO_BATCH_SIZE:-30}"

# Keep the large AdamW state offloaded for the first three-card run.  The
# measured transfer cost was a small part of the step, while correction
# backward is the proven memory peak.  Disable only after a shuffled multi-step
# memory smoke demonstrates enough headroom on every rank.
export ACTOR_FSDP_OPTIMIZER_OFFLOAD="${ACTOR_FSDP_OPTIMIZER_OFFLOAD:-true}"
export ACTOR_BATCH_OFFLOAD="${ACTOR_BATCH_OFFLOAD:-true}"
export REF_FSDP_PARAM_OFFLOAD="${REF_FSDP_PARAM_OFFLOAD:-true}"

# A late-turn search repeats one long state across K=4 probes.  Recompute only
# those EITR probe activations during backward so a rare long shuffled batch
# cannot exceed the per-rank memory ceiling.  This changes compute/memory only;
# cached-old and current probes still share the same deterministic score mode.
export EITR_PROBE_GRADIENT_CHECKPOINTING="${EITR_PROBE_GRADIENT_CHECKPOINTING:-true}"
export EITR_COMPACT_INVALID_STATES="${EITR_COMPACT_INVALID_STATES:-true}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/train_eitr_nq_gate_c_full.sh"
