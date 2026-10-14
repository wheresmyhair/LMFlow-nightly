#!/bin/bash
# Single-GPU smoke test for the SWE-Lego recipe: 8 trajectories, Qwen3-0.6B, one epoch.
# It checks that conversion, rendering, the train/inference checks and training all run.
# Run from the repository root. Override any variable below from the environment.
set -euo pipefail

model_name_or_path=${model_name_or_path:-Qwen/Qwen3-0.6B}
num_samples=${num_samples:-8}
block_size=${block_size:-8192}
data_dir=${data_dir:-data/swe_lego_smoke}
output_dir=${output_dir:-output_models/swe_lego_smoke}

python recipes/agentic/swe_lego/convert.py \
    --source Lego-X/SWE-Lego-Real-Data:resolved \
    --limit "${num_samples}" \
    --output_dir "${data_dir}"

accelerate launch --config_file configs/accelerate_singlegpu_config.yaml \
  examples/finetune.py \
    --model_name_or_path "${model_name_or_path}" \
    --dataset_path "${data_dir}" \
    --conversation_template qwen3_nothink \
    --inference_conversation_template tokenizer \
    --disable_group_texts 1 \
    --block_size "${block_size}" \
    --output_dir "${output_dir}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing 1 \
    --validation_split_percentage 0 \
    --logging_steps 1 \
    --save_strategy no \
    --do_train \
    --report_to none \
    --seed 42
