#!/usr/bin/env bash
set -euo pipefail

# Defaults target the current A800 pilot server and intentionally leave GPU 0 unused.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"

NUM_GPUS="${NUM_GPUS:-2}"
DATA_DIR="${DATA_DIR:-data/nq_search}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
EITR_ENABLED="${EITR_ENABLED:-true}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-eitr-nq-gate-c-smoke}"
WANDB_PROJECT="${WANDB_PROJECT:-EITR-Search-Agent}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-10}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-11}"
TRAIN_DATA_NUM="${TRAIN_DATA_NUM:-32}"
VAL_DATA_NUM="${VAL_DATA_NUM:-64}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-16}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_PROBE_PROMPT_TOKENS="${MAX_PROBE_PROMPT_TOKENS:-$MAX_PROMPT_LENGTH}"
INFORMATIVE_JS_THRESHOLD="${INFORMATIVE_JS_THRESHOLD:-0.01}"
MIN_INFORMATIVE_STATE_RATE="${MIN_INFORMATIVE_STATE_RATE:-0.1}"
SAVE_FREQ="${SAVE_FREQ:--1}"
TEST_FREQ="${TEST_FREQ:-5}"
CHECK_ONLY="${CHECK_ONLY:-false}"

case "$BASE_MODEL" in
    *parallel_search*|*parallel-search*|*finance*)
        echo "Gate C requires a clean single-query Search-R1 initialization, got: $BASE_MODEL" >&2
        exit 2
        ;;
esac

if [[ ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/test.parquet" ]]; then
    echo "Gate C requires $DATA_DIR/train.parquet and $DATA_DIR/test.parquet" >&2
    exit 2
fi

if (( MAX_PROBE_PROMPT_TOKENS < MAX_PROMPT_LENGTH )); then
    echo "MAX_PROBE_PROMPT_TOKENS must be >= MAX_PROMPT_LENGTH for exact same-state probes" >&2
    exit 2
fi

if [[ "$CHECK_ONLY" == "true" ]]; then
    curl --fail --silent --show-error \
        --header 'Content-Type: application/json' \
        --data '{"queries":["who wrote Hamlet"],"topk":3,"return_scores":true}' \
        "$RETRIEVER_URL" >/dev/null
    echo "Gate C preflight passed: data files, clean model path, prompt limit, and retriever are ready."
    exit 0
fi

RAY_TMPDIR="${RAY_TMPDIR:-ray_tmp/eitr_gate_c_smoke}"
mkdir -p "$RAY_TMPDIR"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_data_num="$TRAIN_DATA_NUM" \
    data.val_data_num="$VAL_DATA_NUM" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_batch_size="$VAL_BATCH_SIZE" \
    data.max_start_length=2048 \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length=500 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=false \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size="$PPO_MICRO_BATCH_SIZE" \
    actor_rollout_ref.actor.use_dynamic_bsz=false \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.grad_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.actor.eitr.enabled="$EITR_ENABLED" \
    actor_rollout_ref.actor.eitr.probe_source=online_same_state \
    actor_rollout_ref.actor.eitr.probe_count=4 \
    actor_rollout_ref.actor.eitr.probe_oversample=2 \
    actor_rollout_ref.actor.eitr.max_query_tokens=96 \
    actor_rollout_ref.actor.eitr.probe_seed=20260805 \
    actor_rollout_ref.actor.eitr.max_action_tokens=128 \
    actor_rollout_ref.actor.eitr.max_probe_prompt_tokens="$MAX_PROBE_PROMPT_TOKENS" \
    actor_rollout_ref.actor.eitr.retrieval_score_temperature=0.1 \
    actor_rollout_ref.actor.eitr.min_state_coverage=0.5 \
    actor_rollout_ref.actor.eitr.informative_js_threshold="$INFORMATIVE_JS_THRESHOLD" \
    actor_rollout_ref.actor.eitr.min_informative_state_rate="$MIN_INFORMATIVE_STATE_RATE" \
    actor_rollout_ref.actor.eitr.target_js=0.01 \
    actor_rollout_ref.actor.eitr.initial_beta=0.1 \
    actor_rollout_ref.actor.eitr.dual_lr=0.05 \
    actor_rollout_ref.actor.eitr.beta_max=10.0 \
    actor_rollout_ref.actor.eitr.log_ratio_clip=10.0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=32 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    trainer.logger="['wandb']" \
    +trainer.val_before_train=false \
    +trainer.val_only=false \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="verl_checkpoints/$EXPERIMENT_NAME" \
    max_turns=4 \
    retriever.url="$RETRIEVER_URL" \
    retriever.topk=3 \
    2>&1 | tee "$EXPERIMENT_NAME.log"
