## HOTPOT (4 GPU config) 
# RLVR

nvidia-smi

conda activate rl_new 

wandb online

export WANDB_API_KEY=9799045b86877577494db6f0b896bc7b0e8a1f85

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes 8 --config_file deepspeed.yaml rl_runner.py --config configs/Qwen-3-32B/ambigqa/qwen3_32b_ambigqa_rlcr_single.yaml


## The generation batch size = num_processes * per_device_train_batch_size * gradient_accumulation_steps
## If more gpus are used, training can be sped up by reducing the gradient accumulation steps and increasing num_processes

#bsub -q normal -n 8 -M 80GB -o logs/dec22/qwen3_32b_ambigQA_rlcr_single.out -e logs/dec22/qwen3_32b_ambigQA_rlcr_single.err -gpu "num=8:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB" sh train_runs.sh
