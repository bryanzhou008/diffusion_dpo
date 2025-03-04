#!/usr/bin/env python
# Script to prepare a larger dataset from CSV file for DiffusionDPO training

import os
import csv
import json
import shutil
from pathlib import Path

def prepare_che_dataset(csv_file, output_dir, num_images_per_prompt=9):
    """
    Prepare a larger dataset for DiffusionDPO training from a CSV file.
    
    Args:
        csv_file: Path to the CSV file with dialect/SAE prompt pairs
        output_dir: Output directory for formatted dataset
        num_images_per_prompt: Number of image pairs per prompt (default: 9)
    """
    # Create directory structure
    train_dir = os.path.join(output_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    
    # Prepare metadata
    metadata = []
    total_pairs = 0
    
    # Read CSV file
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            dialect_prompt = row['Dialect_Prompt']
            sae_prompt = row['SAE_Prompt']
            
            # Define source directories for images
            dialect_images_dir = os.path.join(
                "/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/data/image/basic/che/flux.1-dev/dialect_imgs", 
                dialect_prompt
            )
            sae_images_dir = os.path.join(
                "/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/data/image/basic/che/flux.1-dev/sae_imgs", 
                sae_prompt
            )
            
            # Check if directories exist
            if not os.path.exists(dialect_images_dir):
                print(f"Warning: Directory not found: {dialect_images_dir}")
                continue
                
            if not os.path.exists(sae_images_dir):
                print(f"Warning: Directory not found: {sae_images_dir}")
                continue
                
            print(f"Processing prompt pair {row_idx+1}: '{dialect_prompt}' vs '{sae_prompt}'")
            
            # Process each image pair
            for img_idx in range(num_images_per_prompt):
                dialect_img_path = os.path.join(dialect_images_dir, f"{img_idx}.jpg")
                sae_img_path = os.path.join(sae_images_dir, f"{img_idx}.jpg")
                
                # Check if files exist
                if not os.path.exists(dialect_img_path) or not os.path.exists(sae_img_path):
                    print(f"  Warning: Missing image {img_idx} for this prompt, skipping")
                    continue
                
                # Create unique filenames to avoid duplicates across prompts
                win_filename = f"win_{row_idx}_{img_idx}.jpg"
                lose_filename = f"lose_{row_idx}_{img_idx}.jpg"
                
                # Define destination paths
                win_dest = os.path.join(train_dir, win_filename)
                lose_dest = os.path.join(train_dir, lose_filename)
                
                # Copy images
                shutil.copy(sae_img_path, win_dest)
                shutil.copy(dialect_img_path, lose_dest)
                
                # Create metadata entries
                # Entry for the winning image (SAE)
                win_entry = {
                    "file_name": win_filename,
                    "jpg_0": win_filename,  # win image is jpg_0
                    "jpg_1": lose_filename,  # lose image is jpg_1
                    "label_0": 1,  # 1 means jpg_0 (win) is preferred
                    "caption": dialect_prompt  # Use dialect prompt as caption
                }
                
                # Entry for the losing image (dialect)
                lose_entry = {
                    "file_name": lose_filename,
                    "jpg_0": win_filename,
                    "jpg_1": lose_filename,
                    "label_0": 1,
                    "caption": dialect_prompt
                }
                
                metadata.append(win_entry)
                metadata.append(lose_entry)
                total_pairs += 1
    
    # Write metadata.jsonl file
    metadata_file = os.path.join(train_dir, "metadata.jsonl")
    with open(metadata_file, "w") as f:
        for entry in metadata:
            f.write(json.dumps(entry) + "\n")
    
    print(f"Dataset created at {output_dir}")
    print(f"Total prompt pairs processed: {row_idx+1}")
    print(f"Total image pairs created: {total_pairs}")
    print(f"Total metadata entries: {len(metadata)}")
    
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
  --output_dir="tmp-sd15-che"
"""
    
    script_path = os.path.join(output_dir, "run_training.sh")
    with open(script_path, "w") as f:
        f.write(shell_script)
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    print(f"Training script created at {script_path}")

if __name__ == "__main__":
    # Define paths
    csv_file = "/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/data/text/basic/che.csv"
    output_dir = "/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/mitigation/baselines/diffusion_dpo/che-data"
    
    prepare_che_dataset(csv_file, output_dir)