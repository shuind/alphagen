#!/usr/bin/env bash
set -euo pipefail

SEED="${SEED:-0}"
POOL="${POOL:-50}"
CODE="${CODE:-csi300}"
STEPS="${STEPS:-200000}"
LAMBDA_RI="${LAMBDA_RI:-0.0}"
REWARD_PER_STEP="${REWARD_PER_STEP:-0.0}"
LOGDIR="${LOGDIR:-/kaggle/working/runs}"
CKPT_DIR="${CKPT_DIR:-/kaggle/working/checkpoints}"
TB_DIR="${TB_DIR:-/kaggle/working/tb_log}"

BACKBONES=("lstm" "transformer")
REWARD_MODES=("re" "re+func" "re+struct" "re+reg" "re+func+struct" "re+all")

timestamp="$(date +%Y%m%d%H%M%S)"

for backbone in "${BACKBONES[@]}"; do
  for reward_mode in "${REWARD_MODES[@]}"; do
    run_name="bgroup_${backbone}_${reward_mode}_${SEED}_${POOL}_${CODE}_${timestamp}"
    python train_maskable_ppo.py \
      --seed "${SEED}" \
      --pool "${POOL}" \
      --code "${CODE}" \
      --step "${STEPS}" \
      --backbone "${backbone}" \
      --reward_mode "${reward_mode}" \
      --lambda_ri "${LAMBDA_RI}" \
      --reward_per_step "${REWARD_PER_STEP}" \
      --run_name "${run_name}" \
      --logdir "${LOGDIR}" \
      --ckpt_dir "${CKPT_DIR}" \
      --tb_dir "${TB_DIR}"
  done
done
