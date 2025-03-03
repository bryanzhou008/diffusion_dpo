#!/usr/bin/env python
# Script to prepare local data for DiffusionDPO training with direct file placement

import os
import json
import shutil
from pathlib import Path

def prepare_data(wins_source_dir, loses_source_dir, output_dir, prompt, num_pairs=9):
    """
    Prepare local data for DiffusionDPO training with images directly in train directory.
    
    Args:
        wins_source_dir: Directory containing preferred images
        loses_source_dir: Directory containing non-preferred images
        output_dir: Output directory for formatted dataset
        prompt: Text prompt for all image pairs
        num_pairs: Number of image pairs to include
    """
    # Create directory structure without 'images' subdirectory
    train_dir = os.path.join(output_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    
    # Prepare metadata
    metadata = []
    
    for i in range(num_pairs):
        # Define source file paths
        win_img = os.path.join(wins_source_dir, f"{i}.jpg")
        lose_img = os.path.join(loses_source_dir, f"{i}.jpg")
        
        # Check if files exist
        if not os.path.exists(win_img) or not os.path.exists(lose_img):
            print(f"Warning: Missing files for pair {i}, skipping")
            continue
        
        # Define destination paths - save directly in train dir
        win_dest = os.path.join(train_dir, f"win_{i}.jpg")
        lose_dest = os.path.join(train_dir, f"lose_{i}.jpg")
        
        # Copy images
        shutil.copy(win_img, win_dest)
        shutil.copy(lose_img, lose_dest)
        
        # Create metadata entry for win image
        win_entry = {
            "file_name": f"win_{i}.jpg",  # Just the filename, no directory prefix
            "jpg_0": f"win_{i}.jpg",
            "jpg_1": f"lose_{i}.jpg",
            "label_0": 1,
            "caption": prompt
        }
        
        # Create metadata entry for lose image
        lose_entry = {
            "file_name": f"lose_{i}.jpg",  # Just the filename, no directory prefix
            "jpg_0": f"win_{i}.jpg",
            "jpg_1": f"lose_{i}.jpg",
            "label_0": 1,
            "caption": prompt
        }
        
        metadata.append(win_entry)
        metadata.append(lose_entry)
    
    # Write metadata.jsonl file
    metadata_file = os.path.join(train_dir, "metadata.jsonl")
    with open(metadata_file, "w") as f:
        for entry in metadata:
            f.write(json.dumps(entry) + "\n")
    
    print(f"Dataset created at {output_dir}")
    print(f"Total images processed: {len(metadata)}")
    
    # Create the updated shell script
    shell_script = f"""#!/bin/bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="{output_dir}"

accelerate launch train.py \\
  --pretrained_model_name_or_path=$MODEL_NAME \\
  --train_data_dir=$DATA_DIR \\
  --train_batch_size=1 \\
  --dataloader_num_workers=16 \\
  --gradient_accumulation_steps=128 \\
  --max_train_steps=2000 \\
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \\
  --learning_rate=1e-8 --scale_lr \\
  --cache_dir="/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/mitigation/baselines/diffusion_dpo/temp_cache/" \\
  --checkpointing_steps 500 \\
  --beta_dpo 5000 \\
  --output_dir="tmp-sd15-dialect"
"""
    
    script_path = os.path.join(output_dir, "run_training.sh")
    with open(script_path, "w") as f:
        f.write(shell_script)
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    print(f"Training script created at {script_path}")

if __name__ == "__main__":
    # Default paths
    wins_source_dir = "/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/data/image/basic/che/stable-diffusion-3.5-large/sae_imgs/a girl at a school"
    loses_source_dir = "/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/data/image/basic/che/stable-diffusion-3.5-large/dialect_imgs/a chamaca at a school"
    output_dir = "/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/mitigation/baselines/diffusion_dpo/dialect_data_chamaca"
    prompt = "a chamaca at a school"
    
    prepare_data(wins_source_dir, loses_source_dir, output_dir, prompt)