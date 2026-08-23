#!/usr/bin/env bash
set -euo pipefail

# No-update CA-ECAD V11.1 Phase-2 collection.
# This script intentionally collects grouped rollouts only: no old log-prob,
# reference policy, critic, advantage, actor update, validation, or checkpoint.

: "${STORAGE_ROOT:?Set STORAGE_ROOT to a writable data-disk directory.}"
: "${BASE_MODEL:?Set BASE_MODEL to the checkpoint to diagnose.}"

NUM_GPUS="${NUM_GPUS:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
N_AGENT="${N_AGENT:-5}"
CALIBRATION_PROMPTS="${CALIBRATION_PROMPTS:-256}"
DIAGNOSTICS_OUTPUT_DIR="${DIAGNOSTICS_OUTPUT_DIR:-${STORAGE_ROOT}/results/ca_ecad_phase2}"
DATA_DIR="${DATA_DIR:-${STORAGE_ROOT}/data/nq_hotpotqa_train}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train.parquet}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/test.parquet}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
WANDB_PROJECT="${WANDB_PROJECT:-CA-ECAD-Search-Agent}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-ca-ecad-v11-1-phase2-no-update}"

if (( CALIBRATION_PROMPTS % TRAIN_BATCH_SIZE != 0 )); then
  echo "CALIBRATION_PROMPTS must be divisible by TRAIN_BATCH_SIZE" >&2
  exit 2
fi
if (( TRAIN_BATCH_SIZE % NUM_GPUS != 0 )); then
  echo "TRAIN_BATCH_SIZE must be divisible by NUM_GPUS" >&2
  exit 2
fi
if (( N_AGENT < 2 )); then
  echo "N_AGENT must be at least 2 to form peer rollouts" >&2
  exit 2
fi
if [[ -e "${DIAGNOSTICS_OUTPUT_DIR}/phase2_analysis.json" ]]; then
  echo "Refusing to overwrite existing Phase-2 analysis: ${DIAGNOSTICS_OUTPUT_DIR}" >&2
  exit 2
fi

mkdir -p "${STORAGE_ROOT}/ray" "${STORAGE_ROOT}/hydra" "${STORAGE_ROOT}/wandb" \
  "${STORAGE_ROOT}/tmp" "${DIAGNOSTICS_OUTPUT_DIR}"
export RAY_TMPDIR="${STORAGE_ROOT}/ray"
export TMPDIR="${STORAGE_ROOT}/tmp"
export WANDB_DIR="${STORAGE_ROOT}/wandb"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

DIAGNOSTIC_STEPS=$(( CALIBRATION_PROMPTS / TRAIN_BATCH_SIZE ))

python3 -m verl.trainer.main_ppo \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_data_num="${CALIBRATION_PROMPTS}" \
  data.val_data_num=1 \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size=1 \
  data.max_prompt_length=8192 \
  data.max_response_length=500 \
  data.max_start_length=2048 \
  data.max_obs_length=1024 \
  data.shuffle_train_dataloader=false \
  algorithm.adv_estimator=grpo \
  algorithm.ca_ecad.diagnostics_only=true \
  algorithm.ca_ecad.diagnostics_max_steps="${DIAGNOSTIC_STEPS}" \
  algorithm.ca_ecad.diagnostics_output_dir="${DIAGNOSTICS_OUTPUT_DIR}" \
  algorithm.ca_ecad.min_calibration_prompts="${CALIBRATION_PROMPTS}" \
  actor_rollout_ref.model.path="${BASE_MODEL}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.grad_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.rollout.log_prob_micro_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.n_agent="${N_AGENT}" \
  actor_rollout_ref.rollout.temperature=1 \
  actor_rollout_ref.actor.state_masking=true \
  reward_model.enable=false \
  trainer.logger="['wandb']" \
  +trainer.val_before_train=false \
  trainer.n_gpus_per_node="${NUM_GPUS}" \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.project_name="${WANDB_PROJECT}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.total_epochs=1 \
  trainer.total_training_steps="${DIAGNOSTIC_STEPS}" \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir="${STORAGE_ROOT}/checkpoints/${EXPERIMENT_NAME}" \
  hydra.run.dir="${STORAGE_ROOT}/hydra/${EXPERIMENT_NAME}" \
  max_turns=4 \
  retriever.url="${RETRIEVER_URL}" \
  retriever.topk=3

python3 scripts/diagnostics/analyze_ca_ecad_phase2.py \
  --input-dir "${DIAGNOSTICS_OUTPUT_DIR}" \
  --min-calibration-prompts "${CALIBRATION_PROMPTS}" \
  --alpha 2.0 \
  --eta 0.25 \
  --kappa 1.0
