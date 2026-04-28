import torch
from diffusers import StableDiffusionPipeline
import os

# Setup
device = "cuda"
model_id = "runwayml/stable-diffusion-v1-5"
ref_dir = "assignment2/data/part3_ref"
os.makedirs(ref_dir, exist_ok=True)

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

# Ukiyo-e style prompts
prompts = [
    "A majestic ukiyo-e woodblock print of a great wave with Mount Fuji in the background, vibrant colors.",
    "A ukiyo-e portrait of a samurai in traditional armor, detailed line art, muted tones.",
    "A cherry blossom tree in full bloom, ukiyo-e style, artistic composition.",
    "A serene zen garden with rocks and raked sand, ukiyo-e woodblock aesthetic.",
    "A traditional Japanese temple during winter with snow falling, ukiyo-e style.",
    "A geisha walking through a historic Kyoto street at night, ukiyo-e aesthetic.",
    "A crane flying over a pine forest, ukiyo-e woodblock print style.",
    "A koi fish swimming in a pond with lily pads, ukiyo-e style.",
    "A bustling market scene in Edo period Japan, ukiyo-e style.",
    "A moonlit landscape of a bamboo forest, ukiyo-e woodblock aesthetic."
]

print("Generating reference style images...")
for i, prompt in enumerate(prompts):
    generator = torch.Generator(device).manual_seed(1000 + i)
    image = pipe(prompt, num_inference_steps=50).images[0]
    image.save(os.path.join(ref_dir, f"ukiyo_e_{i}.png"))
print(f"Saved 10 reference images to {ref_dir}")
