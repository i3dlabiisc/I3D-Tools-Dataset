import torch
import numpy as np
import os
import logging
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from transformers import DPTImageProcessor, DPTForDepthEstimation

# === PGF/LaTeX styling ===
# mpl.use('pgf')
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.weight": "bold",
    "axes.labelweight": "bold",
    # "axes.labelsize": 14,
    # "axes.titlesize": 14,
    # "legend.fontsize": 10,
    # "legend.title_fontsize": 12,
    # "xtick.labelsize": 20,
    "ytick.labelsize": 24,
    # "xtick.major.width": 1.5,
    # "ytick.major.width": 1.5,
    "figure.dpi": 300,
})

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# === MODEL SETUP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
depth_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
depth_model     = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device).eval()

# === HELPERS ===
def load_image(path):
    img = Image.open(path).convert("RGB")
    inp = depth_processor(images=img, return_tensors="pt").to(device)
    return img, inp

def get_depth_map(inputs):
    with torch.no_grad():
        depth = depth_model(**inputs).predicted_depth.squeeze().cpu().numpy()
    return (depth - depth.min())/(depth.max() - depth.min())

def depth_to_rgb(depth_norm):
    from matplotlib import cm
    rgb = cm.inferno(depth_norm)[:,:,:3]
    return (rgb * 255).astype(np.uint8)

def save_depth_colormap(depth_norm, out_path, size=1024, quality=95):
    img = Image.fromarray(depth_to_rgb(depth_norm))
    img = img.resize((size,size), Image.LANCZOS)
    img.save(out_path, format='JPEG', quality=quality, optimize=True)

def load_mask(mask_path, target_shape):
    """Load a binary mask, resize to target_shape=(H,W), return boolean array."""
    m = Image.open(mask_path).convert("L").resize(target_shape[::-1], Image.NEAREST)
    arr = np.array(m)
    return arr > 127

def evaluate_mse(a, b):
    return np.mean((a.flatten() - b.flatten())**2)

def evaluate_masked_mse(a, b, mask):
    return np.mean((a[mask] - b[mask])**2)

def plot_images_and_depths(images, depths, titles=None, out_path=None):
    fig, axs = plt.subplots(1, 6, figsize=(18, 6), constrained_layout=True)
    title_positions = [(0.155, 0.73), (0.455, 0.73), (0.76, 0.73)]
    for i in range(3):
        img_ax = axs[2 * i]
        dep_ax = axs[2 * i + 1]

        img_ax.imshow(images[i])
        img_ax.axis('off')
        img_ax.set_aspect('equal')

        im = dep_ax.imshow(depths[i], cmap='inferno')
        dep_ax.axis('off')
        dep_ax.set_aspect('equal')

    for i in range(3):
        x, y = title_positions[i]
        fig.text(x, y, rf"\textbf{{{titles[i]}}}", ha='center', va='bottom', fontsize=24)

    # Use last depth map for colorbar to avoid overlaying
    cbar = fig.colorbar(im, ax=axs[1::2], fraction=0.009, pad=0.02)

    pos = cbar.ax.get_position()
    cbar.ax.set_position([pos.x0, pos.y0 + 0.05, pos.width, pos.height * 0.9])  # tweak y0 and height


    cbar.set_label(r"\textbf{Normalized Depth}", weight='bold', labelpad=30, fontsize=24)
    cbar.ax.yaxis.label.set_verticalalignment('bottom')
    cbar.ax.yaxis.label.set_position((0, 0.55))  # (x=0, y=1.05) pushes label slightly up

    plt.savefig(out_path, bbox_inches='tight', dpi=300)  
    # jpg_path = out_path.rsplit('.', 1)[0] + ".jpg"
    # plt.savefig(jpg_path, bbox_inches='tight', dpi=300)

# === MAIN ===
if __name__ == "__main__":
    image_paths = [
        "../../data/depth_test_images/screw/screw_original.jpg",
        "../../data/depth_test_images/screw/screw_composited.jpg",
        "../../data/depth_test_images/screw/screw_refined.jpg",
    ]
    mask_path = "../../data/depth_test_images/screw/screw_original_mask.png"

    out_dir = "../../data/figures/"
    os.makedirs(out_dir, exist_ok=True)

    # 1) compute depths
    images, depths = [], []
    for p in image_paths:
        img, inp = load_image(p)
        images.append(img)
        depths.append(get_depth_map(inp))

    # 2) save colormap JPEGs
    for orig, d in zip(image_paths, depths):
        base = os.path.splitext(os.path.basename(orig))[0]
        out_jpg = os.path.join(out_dir, f"{base}_1024_inf.jpg")
        save_depth_colormap(d, out_jpg)
        print(f"Saved 1024x1024 colormap: {out_jpg}")

    # 3) plot all three depth maps into a single LaTeX-styled PDF
    titles = ["Ground Truth", "Composited", "Harmonized and Diffused"]
    plot_images_and_depths(
        images, depths, titles,
        out_path=os.path.join(out_dir, "depth_maps.pdf")
    )
    print("Saved combined figure to depth_maps.pdf")

    # 4) evaluate MSEs
    gt = depths[0]
    # prepare mask once
    mask = load_mask(mask_path, gt.shape)

    print("\n📊 MSE results:")
    for orig, d in zip(image_paths[1:], depths[1:]):
        mse = evaluate_mse(d, gt)
        mmse = evaluate_masked_mse(d, gt, mask)
        name = os.path.basename(orig)
        print(f"  {name} → MSE: {mse:.3f}, Masked MSE: {mmse:.3f}")
