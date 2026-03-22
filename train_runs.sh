#!/bin/bash
# Example training script for RLCR on the medical differential diagnosis dataset.
#
# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 bash train_runs.sh
#
# The generation batch size = num_processes * per_device_train_batch_size * gradient_accumulation_steps.
# Keep this constant or increase it if more compute is available; lowering it may cause training instability.

# --- RLCR (multi-answer, Brier reward) ---
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes 4 \
    --config_file deepspeed.yaml \
    rl_runner.py \
    --config /dccstor/ishapuri/new_int_inf/cleanrl_medical_copy/configs/Qwen3-8B/rlcr_multi.yaml
# --- RLVR baseline (single-answer) ---
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
#     --num_processes 8 \
#     --config_file deepspeed.yaml \
#     rl_runner.py \
#     --config configs/Qwen3-8B/medical-restart/baselines/2_rlvr_single_standard.yaml
