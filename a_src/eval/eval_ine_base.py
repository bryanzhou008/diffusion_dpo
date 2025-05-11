import os
import torch
from diffusers import StableDiffusionPipeline
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
OUTPUT_DIR = "/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/mitigation/baselines/diffusion_dpo/a_outputs/ine-dpo-basic-4-1-1-9-SAE"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CSV file with test prompts
DATA_CSV = "/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/data/text/train_val_test/4-1-1/basic/ine/test.csv"

# General quality evaluation prompts
general_prompts = [
    "A giant dinosaur frozen into a glacier and recently discovered by scientists, cinematic still",
    "A cute puppy leading a session of the United Nations, newspaper photography",
    "A towering hurricane of rainbow colors towering over a city, cinematic digital art",
    "A redwood tree rising up out of the ocean"
]

# Read the CSV file (it has a header with at least: Dialect_Word, SAE_Word, Dialect_Prompt, SAE_Prompt)
df = pd.read_csv(DATA_CSV, encoding="utf-8")

# Initialize the VQA scorer (for evaluation)
scorer = t2v_metrics.VQAScore(model='clip-flant5-xxl')

# -------------------- Helper Functions --------------------
def load_pipeline(device):
    """
    Load the base Stable Diffusion pipeline on the specified device.
    """
    pipe = StableDiffusionPipeline.from_pretrained(base_model_name, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    pipe.safety_checker = None  # disable safety checker if desired
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
        if hasattr(score, "item"):
            score = score.item()
        score = round(float(score), 4)
        scores.append(score)
    avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0
    return avg_score, scores

# -------------------- Main Processing --------------------
# Determine available GPUs and select one device
num_gpus = torch.cuda.device_count()
if num_gpus == 0:
    raise RuntimeError("No CUDA devices available!")
device = "cuda:0"
print(f"Using device: {device}")

# Load the base Stable Diffusion pipeline
pipe = load_pipeline(device)

# Prepare output file paths for detailed results and summary results.
eval_results_file = os.path.join(OUTPUT_DIR, "evaluation_results.csv")
eval_summary_file = os.path.join(OUTPUT_DIR, "evaluation_results_summary.csv")

# Prepare a list to collect summary results.
summary_results = []

# Open the detailed results file for writing.
with open(eval_results_file, "w") as f_eval:
    # Write header for detailed results.
    f_eval.write("Model,Evaluation_Type,Prompt,Reference_Prompt,Avg_Score\n")
    
    # ---------------------- Dialect Understanding Evaluation ----------------------
    dialect_scores_all = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Dialect Evaluation"):
        dialect_prompt = row["Dialect_Prompt"]
        sae_prompt = row["SAE_Prompt"]
        # Create a folder for this row's output
        row_folder = os.path.join(OUTPUT_DIR, "checkpoint_000", "dialect", dialect_prompt)
        img_paths = generate_and_save_images(pipe, sae_prompt, row_folder, base_seed=100)
        avg_score, _ = evaluate_images(img_paths, sae_prompt)
        dialect_scores_all.append(avg_score)
        line = f"checkpoint_000,dialect_understanding,\"{dialect_prompt}\",\"{sae_prompt}\",{avg_score}\n"
        f_eval.write(line)
    f_eval.write("\n")
    
    # ---------------------- General Quality Evaluation ----------------------
    general_scores_all = []
    for prompt in general_prompts:
        safe_prompt = prompt.replace(" ", "_")[:50]
        row_folder = os.path.join(OUTPUT_DIR, "checkpoint_000", "general", safe_prompt)
        img_paths = generate_and_save_images(pipe, prompt, row_folder, base_seed=300)
        avg_score, _ = evaluate_images(img_paths, prompt)
        general_scores_all.append(avg_score)
        line = f"checkpoint_000,general_quality,\"{prompt}\",\"{prompt}\",{avg_score}\n"
        f_eval.write(line)
    f_eval.write("\n")
    
    # Compute overall average scores.
    avg_dialect_overall = round(sum(dialect_scores_all) / len(dialect_scores_all), 4) if dialect_scores_all else 0.0
    avg_general_overall = round(sum(general_scores_all) / len(general_scores_all), 4) if general_scores_all else 0.0
    
    summary_results.append({
        "Model": "checkpoint_000",
        "Dialect_Understanding_Avg": avg_dialect_overall,
        "General_Quality_Avg": avg_general_overall
    })

# Write the summary CSV file.
summary_df = pd.DataFrame(summary_results)
summary_df.to_csv(eval_summary_file, index=False)

print(f"\nEvaluation complete. Detailed results saved to {eval_results_file}")
print(f"Summary results saved to {eval_summary_file}")
