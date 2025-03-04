import os
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from PIL import Image
import pandas as pd
from tqdm import tqdm
import t2v_metrics

# -------------------- Configuration --------------------
device = 'cuda:7'
torch.set_grad_enabled(False)

# Base model details.
base_model_name = "runwayml/stable-diffusion-v1-5"
num_images = 9
guidance_scale = 7.5
OUTPUT_DIR = "/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/mitigation/baselines/diffusion_dpo/angpow_out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define models: For the base model, ckpt is None.
model_dict = {
    "Orig_SD15": None,
    "DPO_SD15_lr1e-8": "/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/mitigation/baselines/diffusion_dpo/tmp-sd15-dialect/checkpoint-2000-dpo",
    "DPO_SD15_lr1e-7": "/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/mitigation/baselines/diffusion_dpo/tmp-sd15-dialect-dpo-lr1e-7/checkpoint-2000",
    "SFT_SD15": "/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/mitigation/baselines/diffusion_dpo/tmp-sd15-dialect-sft/checkpoint-2000"
}

# Prompts for evaluation.
dialect_prompt = "two ang pows on a table"
sae_prompt = "two red packets on a table"
general_prompts = [
    "A giant dinosaur frozen into a glacier and recently discovered by scientists, cinematic still",
    "A purple raven flying over big sur, light fog, deep focus+closeup, hyper-realistic, volumetric lighting, dramatic lighting, beautiful composition, intricate details, instagram, trending, photograph, film grain and noise, 8K, cinematic, post-production",
    "A cute puppy leading a session of the United Nations, newspaper photography",
    "A pile of sand swirling in the wind forming the shape of a dancer",
    "A smiling beautiful sorceress with long dark hair and closed eyes wearing a dark top surrounded by glowing fire sparks at night, magical light fog, deep focus+closeup, hyper-realistic, volumetric lighting, dramatic lighting, beautiful composition, intricate details, instagram, trending, photograph, film grain and noise, 8K, cinematic, post-production",
    "A redwood tree rising up out of the ocean"
]

# Initialize the VQA scorer.
scorer = t2v_metrics.VQAScore(model='clip-flant5-xxl')

def generate_and_save_images(pipe, prompt, output_folder, base_seed=42):
    os.makedirs(output_folder, exist_ok=True)
    image_paths = []
    for i in range(num_images):
        seed = base_seed + i
        generator = torch.Generator(device=device).manual_seed(seed)
        result = pipe(prompt=prompt, guidance_scale=guidance_scale, generator=generator)
        image = result.images[0]
        img_path = os.path.join(output_folder, f"{i}.jpg")
        image.save(img_path)
        image_paths.append(img_path)
    return image_paths

def evaluate_images(image_paths, reference_prompt):
    scores = []
    for img_path in image_paths:
        score_output = scorer(images=[img_path], texts=[reference_prompt])
        try:
            score = score_output[0][0]
        except TypeError:
            score = score_output
        scores.append(score)
    avg_score = sum(scores) / len(scores) if scores else 0
    return avg_score, scores

def load_pipeline(ckpt):
    # Load the base pipeline.
    pipe = StableDiffusionPipeline.from_pretrained(base_model_name, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    pipe.safety_checker = None
    # If a checkpoint is provided, replace the unet.
    if ckpt is not None:
        unet = UNet2DConditionModel.from_pretrained(ckpt, subfolder="unet", torch_dtype=torch.float16).to(device)
        pipe.unet = unet
    return pipe

# To store results.
results = []

# Process each model sequentially.
for model_name, ckpt in model_dict.items():
    print(f"\nProcessing model: {model_name}")
    model_out_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(model_out_dir, exist_ok=True)
    
    # Load pipeline for current model.
    pipe = load_pipeline(ckpt)
    
    # 1. Dialect understanding evaluation.
    dialect_folder = os.path.join(model_out_dir, "dialect_input")
    print(f"Generating images for dialect prompt: '{dialect_prompt}'")
    dialect_image_paths = generate_and_save_images(pipe, dialect_prompt, dialect_folder, base_seed=100)
    avg_dialect_score, dialect_scores = evaluate_images(dialect_image_paths, sae_prompt)
    results.append({
        "model": model_name,
        "evaluation": "dialect_understanding",
        "input_prompt": dialect_prompt,
        "reference_prompt": sae_prompt,
        "avg_score": avg_dialect_score,
        "individual_scores": dialect_scores
    })
    
    # 2. SAE generation evaluation.
    sae_folder = os.path.join(model_out_dir, "sae_input")
    print(f"Generating images for SAE prompt: '{sae_prompt}'")
    sae_image_paths = generate_and_save_images(pipe, sae_prompt, sae_folder, base_seed=200)
    avg_sae_score, sae_scores = evaluate_images(sae_image_paths, sae_prompt)
    results.append({
        "model": model_name,
        "evaluation": "sae_generation",
        "input_prompt": sae_prompt,
        "reference_prompt": sae_prompt,
        "avg_score": avg_sae_score,
        "individual_scores": sae_scores
    })
    
    # 3. General generation quality evaluation.
    for idx, prompt in enumerate(general_prompts):
        general_folder = os.path.join(model_out_dir, f"general_{idx}")
        print(f"Generating images for general prompt: '{prompt}'")
        general_image_paths = generate_and_save_images(pipe, prompt, general_folder, base_seed=300 + idx * 10)
        avg_general_score, general_scores = evaluate_images(general_image_paths, prompt)
        results.append({
            "model": model_name,
            "evaluation": "general_quality",
            "input_prompt": prompt,
            "reference_prompt": prompt,
            "avg_score": avg_general_score.item(),
            "individual_scores": general_scores
        })
    
    # Clean up: delete pipeline and free GPU memory.
    del pipe
    torch.cuda.empty_cache()

# Save evaluation results.
results_df = pd.DataFrame(results)
results_csv_path = os.path.join(OUTPUT_DIR, "evaluation_results.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"\nEvaluation complete. Results saved to {results_csv_path}")
