import os
import random
import json
from glob import glob
from PIL import Image
import numpy as np
import cv2
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────────────────────
#  Paths
# ────────────────────────────────────────────────────────────────────────────────
bg_root      = "../data/images/bg/final"
fg_root      = "../data/images/fg"
output_root  = "../data/images/multi_instance"
os.makedirs(output_root, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────────
#  Utility – simple HSV‑V harmonisation
# ────────────────────────────────────────────────────────────────────────────────
def harmonize_hsv_brightness(fg_rgba, bg_rgba):
    fg_rgb = fg_rgba[..., :3]
    bg_rgb = bg_rgba[..., :3]
    fg_hsv = cv2.cvtColor(fg_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    bg_hsv = cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    scaling = np.clip(np.mean(bg_hsv[..., 2]) / (np.mean(fg_hsv[..., 2]) + 1e-5), 0.70, 1.30)
    fg_hsv[..., 2] = np.clip(fg_hsv[..., 2] * scaling, 0, 255)
    harmonised_rgb = cv2.cvtColor(fg_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    alpha = fg_rgba[..., 3] if fg_rgba.shape[-1] == 4 else np.full(fg_rgba.shape[:2], 255, np.uint8)
    return np.dstack((harmonised_rgb, alpha))

# ────────────────────────────────────────────────────────────────────────────────
#  Load all foregrounds (big/small)
# ────────────────────────────────────────────────────────────────────────────────
def load_foregrounds(size_dir):
    fg = {}
    for cat in sorted(glob(os.path.join(size_dir, "*"))):
        name = os.path.basename(cat)
        fg[name] = []
        for img_path in glob(os.path.join(cat, "*.jpg")):
            base = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(cat, f"{base}_mask.png")
            if os.path.exists(mask_path):
                fg[name].append((img_path, mask_path))
    return fg

fg_big_all   = load_foregrounds(os.path.join(fg_root, "big"))
fg_small_all = load_foregrounds(os.path.join(fg_root, "small"))

# ────────────────────────────────────────────────────────────────────────────────
#  Tool prompt map  +  YOLO class‑id map
# ────────────────────────────────────────────────────────────────────────────────
tool_prompt_map = {
    'ball_bearing':  'a metal ball bearing',
    'gear':          'a steel gear with teeth',
    'hammer':        'a hammer',
    'measuring_tape':'a measuring tape',
    'nail':          'an iron nail',
    'nut':           'a metal nut',
    'oring':         'a rubber O‑ring',
    'plier':         'a pair of pliers',
    'saw':           'a handheld metal saw',
    'scissors':      'a pair of scissors',
    'screw':         'a metal machine screw',
    'screwdriver':   'a screwdriver',
    'spring':        'a coiled metal spring',
    'utility_knife': 'a utility knife',
    'washer':        'a flat metal washer',
    'wrench':        'a wrench'
}

yolo_id_map = {name: idx for idx, name in enumerate([
    'ball_bearing','gear','hammer','measuring_tape','nail','nut','oring','plier',
    'saw','scissors','screw','screwdriver','spring','utility_knife','washer','wrench'
])}

# ────────────────────────────────────────────────────────────────────────────────
#  Main per‑background routine
# ────────────────────────────────────────────────────────────────────────────────
def process_bg_image(task):
    json_path, prompt_name = task
    base_name  = os.path.basename(json_path).replace("_bboxes.json", "")
    prompt_dir = os.path.join(bg_root, prompt_name)
    out_dir    = os.path.join(output_root, prompt_name)
    os.makedirs(out_dir, exist_ok=True)

    bg_path = os.path.join(prompt_dir, f"{base_name}.png")
    if not os.path.exists(bg_path):
        return f"❌ Missing {bg_path}"

    with open(json_path, "r") as f:
        boxes = json.load(f)

    big_boxes   = [b for b in boxes if b["size"] == "big"]
    small_boxes = [b for b in boxes if b["size"] == "small"]

    # --------------------------------------------------------------------------
    #  up to 10 random composites per background
    # --------------------------------------------------------------------------
    for combo_id in range(10):
        used_types     = set()
        selected_big   = []
        selected_small = []

        # choose BIG
        for _ in big_boxes:
            choices = [(t,p) for t,lst in fg_big_all.items() if t not in used_types for p in lst]
            if not choices: break
            tool, pair = random.choice(choices)
            selected_big.append((tool, pair))
            used_types.add(tool)

        # choose SMALL
        for _ in small_boxes:
            choices = [(t,p) for t,lst in fg_small_all.items() if t not in used_types for p in lst]
            if not choices: break
            tool, pair = random.choice(choices)
            selected_small.append((tool, pair))
            used_types.add(tool)

        if len(selected_big) < len(big_boxes) or len(selected_small) < len(small_boxes):
            print(f"⚠️ Skipping combo {combo_id} for {base_name}: insufficient unique tools")
            continue

        # ----------------------------------------------------------------------
        #  Compose
        # ----------------------------------------------------------------------
        bg_img        = Image.open(bg_path).convert("RGBA")
        composite_mask = Image.new("L", bg_img.size, 0)
        ann_lines      = []                                    # YOLO lines

        for box, (tool, (fg_path, mask_path)) in zip(big_boxes+small_boxes, selected_big+selected_small):
            fg_img = Image.open(fg_path).convert("RGBA")
            mask   = Image.open(mask_path).convert("L")

            bw, bh   = int(box["xmax"]-box["xmin"]), int(box["ymax"]-box["ymin"])
            px, py   = int(box["xmin"]), int(box["ymin"])

            fg_res   = fg_img.resize((bw, bh), Image.LANCZOS)
            mask_res = mask.resize((bw, bh), Image.LANCZOS)

            fg_np    = np.array(fg_res)
            bg_crop  = np.array(bg_img.crop((px, py, px+bw, py+bh)))
            fg_np    = harmonize_hsv_brightness(fg_np, bg_crop)
            fg_harm  = Image.fromarray(fg_np)

            bg_img.paste(fg_harm, (px, py), mask_res)
            composite_mask.paste(mask_res, (px, py), mask_res)

            # ---- YOLO annotation line ----
            iw, ih     = bg_img.size
            x_c        = (box["xmin"] + box["xmax"]) / 2 / iw
            y_c        = (box["ymin"] + box["ymax"]) / 2 / ih
            w_norm     = bw / iw
            h_norm     = bh / ih
            class_id   = yolo_id_map[tool]
            ann_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w_norm:.6f} {h_norm:.6f}")

        # ----------------------------------------------------------------------
        #  Save outputs
        # ----------------------------------------------------------------------
        combo = f"{base_name}_combo{combo_id:02d}"
        bg_img.convert("RGB").save(os.path.join(out_dir,  f"{combo}.jpg"), quality=95)
        composite_mask.save(os.path.join(out_dir,  f"{combo}_mask.png"))
        with open(os.path.join(out_dir, f"{combo}.txt"), "w") as af:
            af.write("\n".join(ann_lines))

        # Prompt sentence = just the tool descriptions list
        tools_sorted = list(used_types)
        if len(tools_sorted) > 1:
            p_str = ", ".join(tool_prompt_map.get(t, f"a {t}") for t in tools_sorted[:-1])
            p_str += f", and {tool_prompt_map.get(tools_sorted[-1], f'a {tools_sorted[-1]}')}"
        else:
            p_str = tool_prompt_map.get(tools_sorted[0], f"a {tools_sorted[0]}")

        with open(os.path.join(out_dir, f"{combo}_prompt.txt"), "w") as pf:
            pf.write(p_str)

    return ""

# ────────────────────────────────────────────────────────────────────────────────
#  Multi‑processing driver
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tasks = []
    for prompt_dir in sorted(glob(os.path.join(bg_root, "prompt*"))):
        prompt_name = os.path.basename(prompt_dir)
        json_dir    = os.path.join(prompt_dir, "tiling_multi_json")
        for json_path in glob(os.path.join(json_dir, "*_bboxes.json")):
            tasks.append((json_path, prompt_name))

    print(f"🧠 Using {cpu_count()} CPU cores for {len(tasks)} background images")
    with Pool(cpu_count()) as pool:
        for res in tqdm(pool.imap_unordered(process_bg_image, tasks), total=len(tasks)):
            pass
