import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib as mpl
from matplotlib.lines import Line2D

# === Use PGF backend for native LaTeX output ===
# mpl.use('pgf')

# === CONFIGURATION ===
warnings.filterwarnings("ignore", message="xFormers is not available")
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    # "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.labelsize": 14,
    "axes.titlesize": 0,
    "legend.fontsize": 10,
    "legend.title_fontsize": 12,
    "xtick.labelsize": 0,
    "ytick.labelsize": 0,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
})

# === DEVICE & MODEL SETUP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device).eval()

# === PATHS & SETTINGS ===
target_size      = 518
use_tsne         = True
batch_size       = 2

gt_root          = "../../data/dino_testing_images/fg"
composited_root  = "../../data/images/single_instance_v2"
refined_root     = "../../data/images/single_instance_v2_refined"

# composited_root  = "../../data/images/temp1"
# refined_root     = "../../data/images/temp2"

class_names      = sorted(os.listdir(gt_root))
palette          = sns.color_palette("tab20", len(class_names))

# choose which sources to include:
sources          = ["Ground Truth", "Refined"]      # omit "composited" if you wish
marker_map       = {"Ground Truth": "o", "composited": "s", "Refined": "^"}

# === PREPROCESSOR & BBOX LOADER ===
preprocess = transforms.Compose([
    transforms.Resize((target_size, target_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.2,0.2,0.2])
])

def load_yolo_bbox(txt_path, img_w, img_h):
    with open(txt_path, 'r') as f:
        _, cx, cy, w, h = map(float, f.readline().split())
    xmin = int((cx - w/2) * img_w)
    ymin = int((cy - h/2) * img_h)
    xmax = int((cx + w/2) * img_w)
    ymax = int((cy + h/2) * img_h)
    return xmin, ymin, xmax, ymax

# === 1) COLLECT GT CROPS ===
if "Ground Truth" in sources:
    print("📦 Collecting GT crops...")
    gt_images, gt_labels, gt_sources = [], [], []
    for cid, cname in enumerate(class_names):
        gt_dir = os.path.join(gt_root, cname)
        if not os.path.isdir(gt_dir):
            continue
        all_imgs = [f for f in os.listdir(gt_dir) if f.endswith('.jpg')]
        for fname in tqdm(all_imgs, desc=f"GT: {cname}"):
            img = Image.open(os.path.join(gt_dir, fname)).convert("RGB")
            gt_images.append(img)
            gt_labels.append(cid)
            gt_sources.append("Ground Truth")
else:
    gt_images, gt_labels, gt_sources = [], [], []

# === 2) COLLECT COMPOSITED CROPS ===
if "composited" in sources:
    print("📦 Collecting Composited crops...")
    comp_images, comp_labels, comp_sources = [], [], []
    for cid, cname in enumerate(class_names):
        comp_dir = os.path.join(composited_root, cname)
        if not os.path.isdir(comp_dir):
            continue
        for fname in tqdm(os.listdir(comp_dir), desc=f"Comp: {cname}"):
            if not fname.endswith('.jpg') or '_mask' in fname or '_prompt' in fname:
                continue
            base = os.path.splitext(fname)[0]
            txt  = os.path.join(comp_dir, base + ".txt")
            if not os.path.exists(txt):
                continue
            try:
                img  = Image.open(os.path.join(comp_dir, fname)).convert("RGB")
                bbox = load_yolo_bbox(txt, *img.size)
                comp_images.append(img.crop(bbox))
                comp_labels.append(cid)
                comp_sources.append("composited")
            except Exception as e:
                print(f"[Warning] Composited {cname}/{base}: {e}")
else:
    comp_images, comp_labels, comp_sources = [], [], []

# === 3) COLLECT REFINED CROPS ===
if "Refined" in sources:
    print("📦 Collecting Refined crops...")
    ref_images, ref_labels, ref_sources = [], [], []
    for cid, cname in enumerate(class_names):
        comp_dir = os.path.join(composited_root, cname)
        ref_dir  = os.path.join(refined_root, cname)
        if not os.path.isdir(comp_dir) or not os.path.isdir(ref_dir):
            continue
        for fname in tqdm(os.listdir(comp_dir), desc=f"Refined: {cname}"):
            if not fname.endswith('.jpg') or '_mask' in fname or '_prompt' in fname:
                continue
            base = os.path.splitext(fname)[0]
            txt  = os.path.join(comp_dir, base + ".txt")
            if not os.path.exists(txt):
                continue
            for variant in ["realvis", "refiner"]:
                path = os.path.join(ref_dir, f"{base}_{variant}.jpg")
                if os.path.exists(path):
                    try:
                        img  = Image.open(path).convert("RGB")
                        bbox = load_yolo_bbox(txt, *img.size)
                        ref_images.append(img.crop(bbox))
                        ref_labels.append(cid)
                        ref_sources.append("Refined")
                    except Exception as e:
                        print(f"[Warning] Refined {cname}/{base}_{variant}: {e}")
else:
    ref_images, ref_labels, ref_sources = [], [], []

# === MERGE ALL ===
all_images  = gt_images  + comp_images  + ref_images
all_labels  = gt_labels  + comp_labels  + ref_labels
all_sources = gt_sources + comp_sources + ref_sources

# === 4) BATCH INFERENCE ===
print(f"\n🚀 Running inference on {len(all_images)} images...")
embs = []
for i in tqdm(range(0, len(all_images), batch_size), desc="Embedding"):
    batch = all_images[i : i+batch_size]
    tsr   = torch.stack([preprocess(im) for im in batch]).to(device)
    with torch.no_grad():
        feats = model.forward_features(tsr)['x_norm_clstoken']
    embs.append(feats.cpu().numpy())
embeddings = np.vstack(embs)

# === 5) DIMENSIONALITY REDUCTION ===
print("\n🔎 Reducing dimensionality...")
reducer = TSNE(n_components=2, perplexity=30, random_state=42) if use_tsne else PCA(n_components=2)
reduced = reducer.fit_transform(embeddings)

# === 6) PLOTTING ===
print("📊 Plotting...")
fig, ax = plt.subplots(figsize=(12,8))

# for cid, cname in enumerate(class_names):
#     # Plot everything _except_ GT first
#     for src in [s for s in sources if s!="Ground Truth"]:
#         idxs = [i for i,(l,s_) in enumerate(zip(all_labels, all_sources)) if l==cid and s_==src]
#         if idxs:
#             pts = reduced[idxs]
#             ax.scatter(
#                 pts[:,0], pts[:,1],
#                 c=[palette[cid]],
#                 marker=marker_map[src],
#                 s=60, alpha=1.0,
#                 edgecolor='black', linewidth=0.7
#             )
#     # Then plot GT on top
#     idxs = [i for i,(l,s_) in enumerate(zip(all_labels, all_sources)) if l==cid and s_=="Ground Truth"]
#     if idxs:
#         pts = reduced[idxs]
#         ax.scatter(
#             pts[:,0], pts[:,1],
#             c=[palette[cid]],
#             marker=marker_map["Ground Truth"],
#             s=60, alpha=1.0,
#             edgecolor='black', linewidth=0.7
#         )

# === KDE HEATMAP PER CLASS ===
for cid, cname in enumerate(class_names):
    # Collect reduced points across all sources
    idxs = [i for i, l in enumerate(all_labels) if l == cid]
    if not idxs:
        continue
    pts = reduced[idxs]
    xs, ys = pts[:, 0], pts[:, 1]
    sns.kdeplot(
        x=xs,
        y=ys,
        fill=True,
        bw_adjust=0.9,
        thresh=0.01,
        levels=2,
        alpha=0.7,
        # cmap=sns.light_palette(palette[cid], as_cmap=True),
        color=palette[cid], # <-- solid color fill
        ax=ax
    )

# === PLOT ONLY MEAN POINTS FOR EACH (class, source) COMBINATION ===
for cid, cname in enumerate(class_names):
    for src in sources:
        idxs = [i for i, (l, s_) in enumerate(zip(all_labels, all_sources)) if l == cid and s_ == src]
        if not idxs:
            continue
        pts = reduced[idxs]
        mean_x, mean_y = np.mean(pts[:, 0]), np.mean(pts[:, 1])
        ax.scatter(
            mean_x, mean_y,
            c=[palette[cid]],
            marker=marker_map[src],
            s=300, alpha=1.0,
            edgecolor='black', linewidth=0.7,
            label=f"{cname}-{src}"  # optional: useful if you want a detailed legend
        )

# grid & axes
x_min, x_max = reduced[:,0].min(), reduced[:,0].max()
y_min, y_max = reduced[:,1].min(), reduced[:,1].max()
ax.set_xticks(np.arange(np.floor(x_min)-1, np.ceil(x_max)+1, 10))
ax.set_yticks(np.arange(np.floor(y_min)-1, np.ceil(y_max)+1, 10))
ax.set_axisbelow(True)
ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.4)
ax.set_xticklabels([]); ax.set_yticklabels([])
ax.tick_params(length=0)
ax.set_xlabel("t-SNE Principal Component 1", fontsize=20)
ax.set_ylabel("t-SNE Principal Component 2", fontsize=20)
for sp in ['top','right','left','bottom']:
    ax.spines[sp].set_visible(True)
    ax.spines[sp].set_linewidth(2)

# === SHAPE LEGEND (source types) ===
shape_handles = [
    Line2D([0],[0], marker=marker_map[src], color='gray', linestyle='None',
           markerfacecolor='gray', markersize=16, markeredgecolor='black', markeredgewidth=0.7, label=src.capitalize())
    for src in sources
]
shape_leg = ax.legend(
    handles=shape_handles,
    # title=r"\textbf{Source Type}",
    loc='upper left',
    frameon=True,
    facecolor='white',
    edgecolor='black',
    ncol=len(sources),
    borderpad=0.5,
    columnspacing=0.2,
    handletextpad=0.0,
    fontsize=20
)
ax.add_artist(shape_leg)                # <-- keep this one visible

# === COLOR LEGEND (tool classes) ===
pretty_names = {
    "ball_bearing":"Ball Bearing","gear":"Gear","hammer":"Hammer",
    "measuring_tape":"Measuring Tape","nail":"Nail","nut":"Nut",
    "oring":"O-ring","plier":"Plier","saw":"Saw","scissors":"Scissors",
    "screw":"Screw","screwdriver":"Screwdriver","spring":"Spring",
    "utility_knife":"Utility Knife","washer":"Washer","wrench":"Wrench"
}
color_handles = [
    Line2D([0],[0], marker='s', color=palette[cid], linestyle='None',
           markerfacecolor=palette[cid], markersize=14,
           label=pretty_names.get(cname, cname), markeredgecolor='none')
    for cid,cname in enumerate(class_names)
]

ax.legend(
    handles=color_handles,
    # title=r"\textbf{Tool Class}",
    loc='lower right',
    bbox_to_anchor=(1,0),
    ncol=4,
    columnspacing=0.5,
    labelspacing=0.2,
    handletextpad=0.0,
    borderpad=0.3,
    frameon=True,
    fontsize=20
)

plt.tight_layout()
plt.savefig("../../data/figures/dinov2_plot.pdf", bbox_inches='tight', dpi=300)