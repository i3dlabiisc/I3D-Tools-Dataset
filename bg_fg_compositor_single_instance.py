import os
import random
import json
from glob import glob
from PIL import Image
import numpy as np
import cv2
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# === Directory setup ===
bg_root = "../data/images/bg/selected"
fg_root = "../data/images/temp2"
output_root = "../data/images/single_instance_v3"
os.makedirs(output_root, exist_ok=True)

tool_classes = [
    'ball_bearing', 'gear', 'hammer', 'measuring_tape', 'nail', 'nut', 'oring', 'plier',
    'saw', 'scissors', 'screw', 'screwdriver', 'spring', 'utility_knife', 'washer', 'wrench'
]
tool_to_id = {name: idx for idx, name in enumerate(tool_classes)}

def harmonize_hsv_brightness(fg_rgba, bg_rgba):
    fg_rgb = fg_rgba[..., :3]
    bg_rgb = bg_rgba[..., :3]
    fg_hsv = cv2.cvtColor(fg_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    bg_hsv = cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    fg_v_mean = np.mean(fg_hsv[..., 2])
    bg_v_mean = np.mean(bg_hsv[..., 2])
    scaling_factor = np.clip(bg_v_mean / (fg_v_mean + 1e-5), 0.70, 1.30)
    fg_hsv[..., 2] *= scaling_factor
    fg_hsv[..., 2] = np.clip(fg_hsv[..., 2], 0, 255)
    harmonized_rgb = cv2.cvtColor(fg_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    alpha = fg_rgba[..., 3] if fg_rgba.shape[-1] == 4 else np.full(fg_rgba.shape[:2], 255, dtype=np.uint8)
    return np.dstack((harmonized_rgb, alpha))

def load_foregrounds(size_dir):
    fg_dict = {}
    for cat in sorted(glob(os.path.join(size_dir, "*"))):
        name = os.path.basename(cat)
        fg_dict[name] = []
        for img_path in glob(os.path.join(cat, "*.jpg")):
            base = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(cat, f"{base}_mask.png")
            if os.path.exists(mask_path):
                fg_dict[name].append((img_path, mask_path))
    return fg_dict

# === Load all foregrounds (both big and small) ===
fg_big_all = load_foregrounds(os.path.join(fg_root, "big"))
fg_small_all = load_foregrounds(os.path.join(fg_root, "small"))

def process_bg_image(args):
    json_path, prompt_name, tool_name, fg_path, mask_path = args
    base_name = os.path.basename(json_path).replace("_bboxes.json", "")
    prompt_dir = os.path.join(bg_root, prompt_name)
    out_dir = os.path.join(output_root, tool_name)
    os.makedirs(out_dir, exist_ok=True)
    bg_path = os.path.join(prompt_dir, f"{base_name}.png")
    if not os.path.exists(bg_path):
        return f"❌ Missing {bg_path}"

    with open(json_path, "r") as f:
        boxes = json.load(f)

    prompt_file_path = os.path.join(prompt_dir + ".txt")
    if not os.path.exists(prompt_file_path):
        return f"❌ Missing background prompt: {prompt_file_path}"
    with open(prompt_file_path, "r") as pf:
        base_prompt = pf.read().strip()

    if not boxes:
        return f"⚠️ No box found in {base_name}"

    box = boxes[0]  # Only one box now

    bg_img = Image.open(bg_path).convert("RGBA")
    composite_mask = Image.new("L", bg_img.size, 0)

    fg_img = Image.open(fg_path).convert("RGBA")
    mask = Image.open(mask_path).convert("L")

    box_w = int(box["xmax"] - box["xmin"])
    box_h = int(box["ymax"] - box["ymin"])
    paste_x = int(box["xmin"])
    paste_y = int(box["ymin"])

    fg_resized = fg_img.resize((box_w, box_h), Image.LANCZOS)
    mask_resized = mask.resize((box_w, box_h), Image.LANCZOS)

    fg_np = np.array(fg_resized)
    bg_crop_np = np.array(bg_img.crop((paste_x, paste_y, paste_x + box_w, paste_y + box_h)))
    harmonized = fg_np if fg_np.shape[:2] != bg_crop_np.shape[:2] else harmonize_hsv_brightness(fg_np, bg_crop_np)
    fg_harmonized = Image.fromarray(harmonized)

    bg_img.paste(fg_harmonized, (paste_x, paste_y), mask_resized)
    composite_mask.paste(mask_resized, (paste_x, paste_y), mask_resized)

    # Create segmentation mask
    seg_mask_rgb = Image.new("RGB", composite_mask.size, (0, 0, 0))
    seg_array = np.array(seg_mask_rgb)
    green_fg = np.array([0, 255, 0], dtype=np.uint8)
    seg_array[np.array(composite_mask) > 0] = green_fg
    seg_mask_rgb = Image.fromarray(seg_array)

    combo_name = f"{base_name}_{tool_name}"
    img_out_path = os.path.join(out_dir, f"{combo_name}.jpg")
    mask_out_path = os.path.join(out_dir, f"{combo_name}_mask.png")
    label_out_path = os.path.join(out_dir, f"{combo_name}.txt")

    bg_img.convert("RGB").save(img_out_path, quality=95)
    seg_mask_rgb.save(mask_out_path)

    # YOLO bbox
    img_w, img_h = bg_img.size
    center_x = (paste_x + box_w / 2) / img_w
    center_y = (paste_y + box_h / 2) / img_h
    rel_w = box_w / img_w
    rel_h = box_h / img_h

    class_id = tool_to_id.get(tool_name, 0)  # fallback to 0 if tool_name is missing
    with open(label_out_path, "w") as lf:
        lf.write(f"{class_id} {center_x:.6f} {center_y:.6f} {rel_w:.6f} {rel_h:.6f}\n")

    # if "no tools" in base_prompt:
    #     prompt = base_prompt.replace("no tools", f"a {tool_name} kept on it")
    # else:
    #     prompt = f"{base_prompt.strip()} with a {tool_name} kept on it"

    # prompt = f"a {tool_name}"

    tool_prompt_map = {
        'ball_bearing': 'a metal ball bearing',
        'gear': 'a steel gear with teeth',
        'hammer': "a carpenter's hammer",
        'measuring_tape': 'a measuring tape',
        'nail': 'an iron nail',
        'nut': 'a metal nut',
        'oring': 'a rubber O-ring',
        'plier': 'a pair of pliers',
        'saw': 'a handheld metal saw',
        'scissors': 'a pair of scissors',
        'screw': 'a metal machine screw', 
        'screwdriver': 'a Phillips head screwdriver',
        'spring': 'a coiled metal spring', # change prompt
        'utility_knife': 'a utility knife', # change prompt
        'washer': 'a flat metal washer',
        'wrench': 'an wrench', # change prompt
    }
    prompt = tool_prompt_map.get(tool_name, f"a {tool_name}")

    with open(os.path.join(out_dir, f"{combo_name}_prompt.txt"), "w") as pf:
        pf.write(prompt)

    return ""

if __name__ == "__main__":
    tasks = []
    for prompt_dir in sorted(glob(os.path.join(bg_root, "prompt*"))):
        prompt_name = os.path.basename(prompt_dir)

        for fg_all in [fg_big_all, fg_small_all]:  # load both big/small foregrounds
            for tool_name, instances in fg_all.items():
                if not instances:
                    continue
                # Use tool-specific JSON
                json_dir = os.path.join(prompt_dir, f"tilling_json_{tool_name}")
                if not os.path.isdir(json_dir):
                    continue
                for json_path in sorted(glob(os.path.join(json_dir, "*_bboxes.json"))):
                    fg_path, mask_path = random.choice(instances)
                    tasks.append((json_path, prompt_name, tool_name, fg_path, mask_path))

    print(f"🧠 Using {cpu_count()} CPU cores for {len(tasks)} tool-background pairs")
    with Pool(cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(process_bg_image, tasks), total=len(tasks)):
            pass
