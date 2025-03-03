#!/bin/bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/mitigation/baselines/diffusion_dpo/dialect_data"

python train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --train_batch_size=1 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=128 \
  --max_train_steps=2000 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
  --learning_rate=1e-6 --scale_lr \
  --cache_dir="/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/mitigation/baselines/diffusion_dpo/temp_cache/" \
  --checkpointing_steps 2000 \
  --beta_dpo 5000 \
  --output_dir="tmp-sd15-dialect-dpo-lr1e-6"