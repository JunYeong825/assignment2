import torch
from diffusers import StableDiffusionImg2ImgPipeline
from utils.metrics import Evaluator
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "runwayml/stable-diffusion-v1-5"
output_dir = "assignment2/outputs/part2"
input_dir = "assignment2/data/part2_inputs"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(input_dir, exist_ok=True)

# 1. Initialize Pipeline
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
evaluator = Evaluator(device=device)

# --- Task 2.0: Prepare Test Set (Daytime Landscapes) ---
print("Preparing Test Set...")
from diffusers import StableDiffusionPipeline
text2img_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

num_images = 20
base_prompt = "A beautiful sunny landscape of "
locations = [
    "a green valley with a river", "snowy mountains", "a dense tropical forest", 
    "a calm blue ocean beach", "a rolling countryside with farms", "a rocky desert canyon",
    "a lavender field in bloom", "a waterfall in a forest", "a serene lake at noon",
    "a majestic castle on a hill", "a city skyline in daylight", "a quiet village road",
    "a cherry blossom garden", "a lighthouse by the sea", "a dense redwood forest",
    "a misty highland", "a vineyard in Tuscany", "a futuristic eco-city",
    "a hidden lagoon", "a sunflower field"
]

test_set = []
for i, loc in enumerate(locations):
    prompt = base_prompt + loc
    img_path = os.path.join(input_dir, f"source_{i}.png")
    if not os.path.exists(img_path):
        generator = torch.Generator(device).manual_seed(100 + i)
        image = text2img_pipe(prompt, num_inference_steps=30).images[0]
        image.save(img_path)
    test_set.append({"path": img_path, "location": loc})

del text2img_pipe
torch.cuda.empty_cache()

# --- Task 2.1 & 2.2: Strength Sweep & Evaluation ---
print("Running Task 2.1 & 2.2: Strength Sweep and Evaluation...")
strengths = [0.2, 0.4, 0.6, 0.8, 1.0]
results = {s: {"clip": [], "lpips": []} for s in strengths}

target_prompt_base = "A dark night landscape of "
for s in strengths:
    print(f"Processing strength: {s}")
    for i, item in enumerate(test_set[:5]): # Sweep on a subset for efficiency
        target_prompt = target_prompt_base + item["location"] + ", moonlight, starry sky"
        init_image = Image.open(item["path"]).convert("RGB").resize((512, 512))
        
        generator = torch.Generator(device).manual_seed(42)
        output = pipe(prompt=target_prompt, image=init_image, strength=s, generator=generator).images[0]
        
        out_path = os.path.join(output_dir, f"result_s{s}_i{i}.png")
        output.save(out_path)
        
        results[s]["clip"].append(evaluator.compute_clip_score(out_path, target_prompt))
        results[s]["lpips"].append(evaluator.compute_lpips(item["path"], out_path))

# Visual Grid for One Sample
sample_idx = 0
grid_images = [Image.open(test_set[sample_idx]["path"])]
grid_titles = ["Source"]
for s in strengths:
    grid_images.append(Image.open(os.path.join(output_dir, f"result_s{s}_i{sample_idx}.png")))
    grid_titles.append(f"Strength {s}")

def save_grid(images, rows, cols, filename, titles=None):
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    axes = axes.flatten()
    for i, img in enumerate(images):
        axes[i].imshow(img)
        if titles: axes[i].set_title(titles[i])
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

save_grid(grid_images, 1, len(grid_images), "task2_1_strength_grid.png", grid_titles)

# --- Task 2.3: Failure Case Analysis (Simulated for Script) ---
# We will identify potential failure cases by looking at extreme strength values.
# Case 1: Low strength (0.2) - Image doesn't change enough to reflect "night".
# Case 2: High strength (1.0) - Structural identity of source is lost.
# Case 3: Conflicting prompt (e.g., "sunny night") - Semantic confusion.

# Summary Report
print("\n--- Summary Results Part 2 ---")
for s in strengths:
    m_clip = np.mean(results[s]["clip"])
    m_lpips = np.mean(results[s]["lpips"])
    print(f"Strength {s}: CLIP Score={m_clip:.4f}, LPIPS={m_lpips:.4f}")
