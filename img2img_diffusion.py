import os
import torch
import random
from glob import glob
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline

# === Config ===
input_root = "../data/images/single_instance_v3"
output_root = "../data/images/single_instance_v3/refined"
os.makedirs(output_root, exist_ok=True)

strength_realvis = 0.2
guidance_realvis = 10.0
strength_refiner = 0.2
guidance_refiner = 10.0
max_images_per_tool = 500
use_model = "realvis"  # Options: "realvis", "refiner", or "both"
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Conditional Model Loading ===
realvis_pipe = None
sdxl_refiner_pipe = None

if use_model in ["realvis", "both"]:
    print("⏳ Loading RealVisXL V4.0 model...")
    realvis_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "SG161222/RealVisXL_V4.0",
        torch_dtype=torch.float16
    ).to(device)

if use_model in ["refiner", "both"]:
    print("⏳ Loading SDXL refiner 1.0 model...")
    sdxl_refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float16
    ).to(device)

# === Process each tool folder ===
tool_dirs = sorted([d for d in glob(os.path.join(input_root, "*")) if os.path.isdir(d)])

for folder in tool_dirs:
    images_in_folder = [
        os.path.join(folder, file)
        for file in os.listdir(folder)
        if (
            file.endswith(".jpg")
            and "_mask" not in file
            and not file.endswith("_realvis.jpg")
            and not file.endswith("_base.jpg")
            and not file.endswith("_refiner.jpg")
        )
    ]

    print(f"🔧 Processing {len(images_in_folder)} images in folder: {os.path.basename(folder)}")

    for idx, input_path in enumerate(images_in_folder):
        if idx >= max_images_per_tool:
            print(f"⏭️ Reached limit of {max_images_per_tool} images in {os.path.basename(folder)} — moving to next Tool.")
            break

        base_name = os.path.splitext(os.path.basename(input_path))[0]
        prompt_path = os.path.join(folder, f"{base_name}_prompt.txt")
        rel_subdir = os.path.relpath(folder, input_root)
        output_dir = os.path.join(output_root, rel_subdir)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(prompt_path):
            print(f"⚠️ Skipping {input_path} — prompt file not found.")
            continue

        with open(prompt_path, "r") as f:
            prompt = f.read().strip()

        image = Image.open(input_path).convert("RGB").resize((1024, 1024))
        seed = random.randint(0, 1_000_000)
        generator = torch.manual_seed(seed)
        print(f"🎨 {os.path.basename(folder)}/{base_name} | 📝 {prompt[:60]}... | 🎲 Seed: {seed}")

        # === Run selected models ===
        if use_model in ["realvis", "both"]:
            out_realvis = realvis_pipe(
                prompt=prompt,
                image=image,
                strength=strength_realvis,
                guidance_scale=guidance_realvis,
                generator=generator
            ).images[0]

            out_realvis.convert("RGB").save(
                os.path.join(output_dir, f"{base_name}_realvis.jpg"),
                format="JPEG",
                quality=92
            )

        if use_model in ["refiner", "both"]:
            out_refiner = sdxl_refiner_pipe(
                prompt=prompt,
                image=image,
                strength=strength_refiner,
                guidance_scale=guidance_refiner,
                generator=generator
            ).images[0]

            out_refiner.convert("RGB").save(
                os.path.join(output_dir, f"{base_name}_refiner.jpg"),
                format="JPEG",
                quality=92
            )

        print(f"✅ Saved results for: {base_name} [{use_model}]")

print("🏁 All processing complete.")
