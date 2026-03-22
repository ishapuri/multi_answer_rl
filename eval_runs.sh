#!/bin/bash
# Example evaluation script.
# See eval_configs/ for available evaluation configurations.
#
# Usage: CUDA_VISIBLE_DEVICES=0 bash eval_runs.sh

CUDA_VISIBLE_DEVICES=0 python evaluation.py --config eval_configs/final_medical_results/rlcr_multi_standard.json
