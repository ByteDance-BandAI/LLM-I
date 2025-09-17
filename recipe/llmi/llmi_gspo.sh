#!/usr/bin/env bash
# ------------------------------------------------------------
#  Experiment Metadata
# ------------------------------------------------------------
PROJECT_NAME="LLMI"
EXP_SUFFIX="gspo"
EXP_NAME="Qwen3-30B-Instruct-${EXP_SUFFIX}"
SAVE_DIR="YOUR_PATH/${PROJECT_NAME}/${EXP_NAME}"

# ------------------------------------------------------------
#  Ray & Cluster Settings
# ------------------------------------------------------------
export RAY_ADDRESS="YOUR_RAY_ADDRESS"
NNODES=${NNODES:-2}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# ------------------------------------------------------------
#  API for Diffusion or Search
# ------------------------------------------------------------
# for seedream (you can also use Qwen-Image)
export ARK_API_KEY="YOUR_ARK_API_KEY"
# for Google Search
export SERP_API_KEY="YOUR_SERP_API_KEY"

# ------------------------------------------------------------
#  Judge
# ------------------------------------------------------------
export LLM_JUDGE_BASE_URL="YOUR_LLM_JUDGE_BASE_URL"
export MLLM_JUDGE_BASE_URL="YOUR_MLLM_JUDGE_BASE_URL"

# ------------------------------------------------------------
#  Model & Data Paths
# ------------------------------------------------------------
# Model Path
model_path="Qwen3-30B-A3B-Instruct-2507"
MODEL_PATH="${MODEL_PATH:-${model_path}}"

# Data Path
train_files="['train.parquet']"
test_files="['test.parquet']"
TRAIN_FILE="${TRAIN_FILE:-${train_files}}"
TEST_FILE="${TEST_FILE:-${test_files}}"

# ------------------------------------------------------------
#  Core Algorithm Hyper-parameters
# ------------------------------------------------------------
# Advantage Estimation
loss_mode="gspo"
adv_estimator="grpo"
clip_ratio_low=0.0003
clip_ratio_high=0.0004

# KL Control
use_kl_in_reward=false
kl_coef=0.0
use_kl_loss=false
kl_loss_coef=0.0

# Loss Aggregation
loss_agg_mode="seq-mean-token-mean"

# Sequence Length
max_prompt_length=$((1024 * 8))
max_response_length=$((1024 * 24))

# Batch Size
train_prompt_bsz=16
train_prompt_mini_bsz=16
n_resp_per_prompt=8
max_num_gen_batches=-1
train_micro_bsz_per_gpu=1
log_prob_micro_bsz_per_gpu=1

# Judge
llm_text_factor=1.4
llm_image_factor=0.6
mllm_quality_factor=1.0
mllm_ti_factor=1.0
mllm_tq_factor=1.0
llm_factor=0.5
mllm_factor=0.3
img_num_factor=0.2

# Parallelism
seq_parallel_size=2
gen_tensor_parallel_size=2

# Memory & Efficiency
param_optim_offload=false
use_dynamic_bsz=false
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
enable_grad_ckpt=true

# System Prompt
replace_system_prompt=true
new_sp_path="system_prompts/training/llmi.txt"

# Trainer Schedule & Logging
val_before_train=true
save_freq=10
test_freq=10
total_epochs=1
log_train_generations=8
log_val_generations=8
log_train_freq=20

WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}


# ------------------------------------------------------------
#  Launch Training
# ------------------------------------------------------------
ray job submit --no-wait \
    --working-dir $(pwd) \
    -- python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.shuffle=true \
    data.filter_overlong_prompts=false \
    data.prompt_key="prompt" \
    data.truncation="error" \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    +data.replace_system_prompt=${replace_system_prompt} \
    +data.new_sp_path="${new_sp_path}" \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_bsz_per_gpu} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tensor_parallel_size} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    algorithm.adv_estimator="${adv_estimator}" \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_micro_bsz_per_gpu} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=6 \
    actor_rollout_ref.actor.optim.warmup_style='cosine' \
    actor_rollout_ref.actor.optim.min_lr_ratio=0.01 \
    actor_rollout_ref.actor.optim.num_cycles=0.5 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode="${loss_agg_mode}" \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${param_optim_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${param_optim_offload} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${seq_parallel_size} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${param_optim_offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${seq_parallel_size} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=${enable_grad_ckpt} \
    actor_rollout_ref.model.use_remove_padding=true \
    reward_model.reward_manager="llmi" \
    +reward_model.llm_cfg.llm_text_factor=${llm_text_factor} \
    +reward_model.llm_cfg.llm_image_factor=${llm_image_factor} \
    +reward_model.mllm_cfg.mllm_quality_factor=${mllm_quality_factor} \
    +reward_model.mllm_cfg.mllm_ti_factor=${mllm_ti_factor} \
    +reward_model.mllm_cfg.mllm_tq_factor=${mllm_tq_factor} \
    +reward_model.llm_cfg.llm_factor=${llm_factor} \
    +reward_model.mllm_cfg.mllm_factor=${mllm_factor} \
    +reward_model.img_num_factor=${img_num_factor} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.val_before_train=${val_before_train} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.log_val_generations=${log_val_generations} \
    +trainer.log_train_generations=${log_train_generations} \
    +trainer.log_train_freq=${log_train_freq} \
    trainer.nnodes=${NNODES} \
    trainer.total_epochs=${total_epochs} \
    trainer.default_local_dir="${SAVE_DIR}"
