import torch
from diffusers import (
    StableDiffusionPipeline, 
    DDPMScheduler, 
    DDIMScheduler, 
    EulerDiscreteScheduler, 
    DPMSolverMultistepScheduler
)
from utils.metrics import CLIPScoreEvaluator
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "runwayml/stable-diffusion-v1-5"
output_dir = "assignment2/outputs/part1"
os.makedirs(output_dir, exist_ok=True)

# 1. Initialize Pipeline
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
evaluator = CLIPScoreEvaluator(device=device)

# Tasks Configuration
prompts = [
    "A group of diverse people having a picnic in a sunny park with a basket of fruits and a dog playing.", # Scene
    "A high-detail close-up portrait of an elderly man with wisdom in his eyes, soft studio lighting.", # Portrait
    "An abstract representation of the concept of time, flowing clocks and surreal cosmic energy.", # Abstract
    "A majestic mountain range at sunset with a crystal clear lake in the foreground reflecting the orange sky.", # Landscape
    "A futuristic cyberpunk city street at night with neon signs, rain reflections on the pavement, and flying cars." # Other
]

schedulers = {
    "DDPM": DDPMScheduler.from_config(pipe.scheduler.config),
    "DDIM": DDIMScheduler.from_config(pipe.scheduler.config),
    "Euler": EulerDiscreteScheduler.from_config(pipe.scheduler.config),
    "DPM-Solver": DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
}

def save_grid(images, rows, cols, filename, titles=None):
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    if rows == 1: axes = [axes]
    if cols == 1: axes = [[ax] for ax in axes]
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        axes[r][c].imshow(img)
        if titles: axes[r][c].set_title(titles[i], fontsize=10)
        axes[r][c].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# --- Task 1.1: Scheduler Comparison ---
print("Running Task 1.1: Scheduler Comparison...")
task1_1_results = {}
for scheduler_name, scheduler in schedulers.items():
    pipe.scheduler = scheduler
    task1_1_results[scheduler_name] = []
    for i, prompt in enumerate(prompts):
        generator = torch.Generator(device).manual_seed(42)
        image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, generator=generator).images[0]
        img_path = os.path.join(output_dir, f"task1_1_{scheduler_name}_p{i}.png")
        image.save(img_path)
        score = evaluator.compute_score(img_path, prompt)
        task1_1_results[scheduler_name].append(score)

# Visual Grid for Task 1.1
all_task1_1_images = []
titles = []
for i in range(len(prompts)):
    for s_name in schedulers.keys():
        img = Image.open(os.path.join(output_dir, f"task1_1_{s_name}_p{i}.png"))
        all_task1_1_images.append(img)
        titles.append(f"{s_name} (P{i+1})")
save_grid(all_task1_1_images, len(prompts), len(schedulers), "task1_1_grid.png", titles)

# --- Task 1.2: Step Count vs. Quality ---
print("Running Task 1.2: Step Count vs. Quality...")
pipe.scheduler = schedulers["DPM-Solver"]
step_counts = [5, 10, 20, 30, 50, 100]
landscape_prompt = prompts[3]
task1_2_images = []
task1_2_scores = []
for steps in step_counts:
    generator = torch.Generator(device).manual_seed(42)
    image = pipe(landscape_prompt, num_inference_steps=steps, guidance_scale=7.5, generator=generator).images[0]
    img_path = os.path.join(output_dir, f"task1_2_steps_{steps}.png")
    image.save(img_path)
    score = evaluator.compute_score(img_path, landscape_prompt)
    task1_2_images.append(image)
    task1_2_scores.append(score)
save_grid(task1_2_images, 1, len(step_counts), "task1_2_grid.png", [f"Steps: {s}" for s in step_counts])

# --- Task 1.3: Classifier-Free Guidance Analysis ---
print("Running Task 1.3: CFG Analysis...")
cfg_scales = [1.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0]
portrait_prompt = prompts[1]
task1_3_images = []
task1_3_scores = []
for cfg in cfg_scales:
    generator = torch.Generator(device).manual_seed(42)
    image = pipe(portrait_prompt, num_inference_steps=50, guidance_scale=cfg, generator=generator).images[0]
    img_path = os.path.join(output_dir, f"task1_3_cfg_{cfg}.png")
    image.save(img_path)
    score = evaluator.compute_score(img_path, portrait_prompt)
    task1_3_images.append(image)
    task1_3_scores.append(score)
save_grid(task1_3_images, 1, len(cfg_scales), "task1_3_grid.png", [f"CFG: {c}" for c in cfg_scales])

# --- Output Summary ---
print("\n--- Summary Results ---")
print("Task 1.1: CLIP Scores (Mean ± Std)")
for s_name, scores in task1_1_results.items():
    print(f"{s_name}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

print("\nTask 1.2: CLIP Scores per Step Count")
for s, score in zip(step_counts, task1_2_scores):
    print(f"Steps {s}: {score:.4f}")

print("\nTask 1.3: CLIP Scores per CFG Scale")
for c, score in zip(cfg_scales, task1_3_scores):
    print(f"CFG {c}: {score:.4f}")
