import os
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from PIL import Image
import pandas as pd
from tqdm import tqdm
import t2v_metrics

# -------------------- Configuration --------------------
# Generation settings
num_images = 4
guidance_scale = 7.5
base_model_name = "runwayml/stable-diffusion-v1-5"

# Folder for output
OUTPUT_DIR = "/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/mitigation/baselines/diffusion_dpo/a_outputs/ine-dpo-4-1-1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CSV file with test prompts
DATA_CSV = "/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/data/text/train_val_test/4-1-1/basic/ine/test.csv"

# General quality evaluation prompts
general_prompts = [
    "A giant dinosaur frozen into a glacier and recently discovered by scientists, cinematic still",
    "A purple raven flying over big sur, light fog, deep focus+closeup, hyper-realistic, volumetric lighting, dramatic lighting, beautiful composition, intricate details, instagram, trending, photograph, film grain and noise, 8K, cinematic, post-production",
    "A cute puppy leading a session of the United Nations, newspaper photography",
    "Worm eye view of rocketship",
    "A towering hurricane of rainbow colors towering over a city, cinematic digital art",
    "A redwood tree rising up out of the ocean"
]

# Generate model dictionary using checkpoints from 100 to 2000 (step 100)
checkpoint_nums = list(range(200, 2401, 400))
model_dict = {}
for num in checkpoint_nums:
    model_key = f"checkpoint_{num}"
    model_path = f"/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/mitigation/baselines/diffusion_dpo/a_checkpoints/ine-dpo-2-2-2/checkpoint-{num}"
    model_dict[model_key] = model_path

# Read the CSV file (it has a header with at least: Dialect_Word, SAE_Word, Dialect_Prompt, SAE_Prompt)
df = pd.read_csv(DATA_CSV, encoding="utf-8")

# Initialize the VQA scorer (for evaluation)
scorer = t2v_metrics.VQAScore(model='clip-flant5-xxl')

# -------------------- Helper Functions --------------------
def load_pipeline(ckpt, device):
    """
    Load the base Stable Diffusion pipeline on the specified device.
    If a checkpoint is provided, replace the UNet.
    """
    pipe = StableDiffusionPipeline.from_pretrained(base_model_name, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    pipe.safety_checker = None  # disable safety checker if desired
    if ckpt is not None:
        unet = UNet2DConditionModel.from_pretrained(ckpt, subfolder="unet", torch_dtype=torch.float16).to(device)
        pipe.unet = unet
    return pipe

def generate_and_save_images(pipe, prompt, output_folder, base_seed=42):
    """
    Generate num_images images using the provided prompt and save them in output_folder.
    Returns a list of file paths for the saved images.
    """
    os.makedirs(output_folder, exist_ok=True)
    image_paths = []
    for i in range(num_images):
        seed = base_seed + i
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
        result = pipe(prompt=prompt, guidance_scale=guidance_scale, generator=generator)
        image = result.images[0]
        img_path = os.path.join(output_folder, f"{i}.jpg")
        image.save(img_path)
        image_paths.append(img_path)
    return image_paths

def evaluate_images(image_paths, reference_prompt):
    """
    Evaluate a list of images against the reference_prompt using the VQA scorer.
    Returns the average score (rounded to 4 decimals) and a list of individual scores (also rounded).
    """
    scores = []
    for img_path in image_paths:
        score_output = scorer(images=[img_path], texts=[reference_prompt])
        try:
            score = score_output[0][0]
        except TypeError:
            score = score_output
        # If score is a tensor, convert to float
        if hasattr(score, "item"):
            score = score.item()
        score = round(float(score), 4)
        scores.append(score)
    avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0
    return avg_score, scores

# -------------------- Main Processing --------------------
# Determine available GPUs and assign each model in round-robin
num_gpus = torch.cuda.device_count()
if num_gpus == 0:
    raise RuntimeError("No CUDA devices available!")
print(f"Found {num_gpus} GPUs.")

# Prepare output file paths for detailed results and summary results.
eval_results_file = os.path.join(OUTPUT_DIR, "evaluation_results.csv")
eval_summary_file = os.path.join(OUTPUT_DIR, "evaluation_results_summary.csv")

# Prepare a list to collect summary results for each model.
summary_results = []

# Open the detailed results file for writing.
with open(eval_results_file, "w") as f_eval:
    # Write header for detailed results.
    f_eval.write("Model,Evaluation_Type,Prompt,Reference_Prompt,Avg_Score,Individual_Scores\n")
    
    # Process each model one by one.
    model_keys = list(model_dict.keys())
    for idx, model_name in enumerate(model_keys):
        device = f"cuda:{idx % num_gpus}"
        print(f"\nProcessing model {model_name} on {device}")
        ckpt = model_dict[model_name]
        pipe = load_pipeline(ckpt, device)
        
        # To accumulate per-model scores for summary
        dialect_scores_all = []
        general_scores_all = []
        
        # ---------------------- Dialect Understanding Evaluation ----------------------
        # For each row in the CSV file (each test prompt)
        for i, row in tqdm(df.iterrows(), total=len(df), desc=f"{model_name} dialect eval"):
            dialect_prompt = row["Dialect_Prompt"]
            sae_prompt = row["SAE_Prompt"]
            # Create a folder for this row's output: e.g., OUTPUT_DIR/model_name/dialect/row_i
            row_folder = os.path.join(OUTPUT_DIR, model_name, "dialect", dialect_prompt)
            img_paths = generate_and_save_images(pipe, dialect_prompt, row_folder, base_seed=100)
            avg_score, ind_scores = evaluate_images(img_paths, sae_prompt)
            dialect_scores_all.append(avg_score)
            # Write the detailed result for this row.
            ind_scores_str = ";".join(map(str, ind_scores))
            line = f"{model_name},dialect_understanding,\"{dialect_prompt}\",\"{sae_prompt}\",{avg_score},{ind_scores_str}\n"
            f_eval.write(line)
        
        # Write an empty line after finishing dialect evaluations for this model.
        f_eval.write("\n")
        
        # ---------------------- General Quality Evaluation ----------------------
        # For each fixed general prompt, generate and evaluate images (against the prompt itself)
        for prompt in general_prompts:
            # Create a folder for this general prompt's outputs. Sanitize prompt for folder name.
            safe_prompt = prompt.replace(" ", "_")[:50]
            row_folder = os.path.join(OUTPUT_DIR, model_name, "general", safe_prompt)
            img_paths = generate_and_save_images(pipe, prompt, row_folder, base_seed=300)
            avg_score, ind_scores = evaluate_images(img_paths, prompt)
            general_scores_all.append(avg_score)
            ind_scores_str = ";".join(map(str, ind_scores))
            line = f"{model_name},general_quality,\"{prompt}\",\"{prompt}\",{avg_score},{ind_scores_str}\n"
            f_eval.write(line)
        
        # Write an empty line after finishing general quality evaluations for this model.
        f_eval.write("\n")
        
        # Clean up the model pipeline from GPU memory.
        del pipe
        torch.cuda.empty_cache()
        
        # Compute overall average scores for this model
        avg_dialect_overall = round(sum(dialect_scores_all) / len(dialect_scores_all), 4) if dialect_scores_all else 0.0
        avg_general_overall = round(sum(general_scores_all) / len(general_scores_all), 4) if general_scores_all else 0.0
        
        summary_results.append({
            "Model": model_name,
            "Dialect_Understanding_Avg": avg_dialect_overall,
            "General_Quality_Avg": avg_general_overall
        })

# Write the summary CSV file.
summary_df = pd.DataFrame(summary_results)
summary_df.to_csv(eval_summary_file, index=False)

print(f"\nEvaluation complete. Detailed results saved to {eval_results_file}")
print(f"Summary results saved to {eval_summary_file}")
