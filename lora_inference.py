from diffusers import AutoPipelineForText2Image
import matplotlib.pyplot as plt
import os

checkpoint_dir = "../../data/lora_finetune/trained_checkpoints/gear/sdxl/checkpoint-10000"
output_dir = os.path.join(checkpoint_dir)

pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0").to("cuda")
pipeline.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
pipeline.load_lora_weights(checkpoint_dir, weight_name="pytorch_lora_weights.safetensors")
# Generate 4 images
prompts = [
    "a steel gear with teeth on a wooden table",
    "a steel gear with teeth on a wooden table",
    "a steel gear with teeth on a wooden table",
    "a steel gear with teeth on a wooden table"
]

images = [pipeline(prompt).images[0] for prompt in prompts]

os.makedirs(output_dir, exist_ok=True)
for i, (img, prompt) in enumerate(zip(images, prompts)):
    img_path = os.path.join(output_dir, f"image_{i}.jpg")
    img.save(img_path, format='JPEG')


fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, img, prompt in zip(axes, images, prompts):
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(prompt, fontsize=10)

plt.tight_layout()
plt.show()