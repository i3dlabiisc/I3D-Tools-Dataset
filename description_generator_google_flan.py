import os
import torch
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# === CONFIGURATION ===
TOOLS_DIR    = "../data/images/lora_dataset/temp"
PROMPTS_DIR  = "../data/images/bg/final"
OUT_DIR      = "../data/images/lora_dataset/temp"
MODEL_NAME   = "google/flan-t5-xl" # Seq2Seq model for text generation
BATCH_SIZE   = 64  # Larger batch = faster (tune based on GPU memory)

# === SCENE TEMPLATE ===
SCENE_TEMPLATE = (
    "Instruction: You are a scene-description assistant.\n"
    "Input: A background description of a surface and its surroundings, and a tool phrase.\n"
    "Output: Produce a single, fluent English sentence that:\n"
    "  1. Smoothly integrates the tool phrase into the scene description.\n"
    "  2. Must place the tool above the described surface.\n"
    "  3. Keep the scene description as close to the original as possible.\n"
    "Additional requirement: Every sentence must begin with “A <tool phrase> lying on the described surface…”,\n"
)

# === TOOL NAME MAPPINGS ===
tool_prompt_map = {
    'ball_bearing':    'a metal ball bearing',
    'gear':            'a steel gear with teeth',
    'hammer':          "a hammer",
    'measuring_tape':  'a measuring tape',
    'nail':            'an iron nail',
    'nut':             'a metal nut',
    'oring':           'a rubber O-ring',
    'plier':           'a pair of pliers',
    'saw':             'a handheld metal saw',
    'scissors':        'a pair of scissors',
    'screw':           'a metal screw',
    'screwdriver':     'a screwdriver',
    'spring':          'a coiled metal spring',
    'utility_knife':   'a utility knife',
    'washer':          'a flat metal washer',
    'wrench':          'a wrench',
}

# === UTILS ===
def clean_bg_text(text: str) -> str:
    for phrase in ["no tools", "empty surface", "worn but empty surface"]:
        text = re.sub(re.escape(phrase), "", text, flags=re.IGNORECASE)
    return re.sub(r"[ ,]{2,}", " ", text).strip()

def build_prompt(bg: str, tool_phrase: str) -> str:
    return f"{SCENE_TEMPLATE}\n\nBackground: {bg.strip()}\nTool: {tool_phrase}\n\nAnswer:"

# === LOAD FLAN-T5 PIPELINE ===
device = 0 if torch.cuda.is_available() else -1
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32).to("cuda" if device == 0 else "cpu")

generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device,
    batch_size=BATCH_SIZE,
    max_length=100,
    do_sample=True,
    temperature=0.5,
    top_p=0.9
)

# === COLLECT RECORDS ===
records = []
for tool in sorted(os.listdir(TOOLS_DIR)):
    tool_folder = os.path.join(TOOLS_DIR, tool)
    if not os.path.isdir(tool_folder): continue

    for fname in os.listdir(tool_folder):
        if not fname.lower().endswith((".jpg", ".png")): continue
        prompt_id = fname.split("_", 1)[0]
        bg_path = os.path.join(PROMPTS_DIR, f"{prompt_id}.txt")
        if not os.path.exists(bg_path):
            print(f"[!] Missing: {prompt_id}.txt")
            continue

        bg_text = clean_bg_text(open(bg_path).read())
        tool_phrase = tool_prompt_map.get(tool, f"a {tool}")
        out_fname = os.path.join(tool, os.path.splitext(fname)[0] + ".txt")

        records.append({
            "prompt": build_prompt(bg_text, tool_phrase),
            "out_fname": out_fname
        })

# === BATCH GENERATION ===
print(f"[INFO] Total prompts to generate: {len(records)}")
all_outputs = []
num_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE

for i in tqdm(range(num_batches), desc="Generating prompts"):
    batch = records[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
    inputs = [r["prompt"] for r in batch]
    outs = generator(inputs, truncation=True)
    all_outputs.extend([out["generated_text"].strip() for out in outs])

# === SAVE OUTPUTS ===
for rec, text in tqdm(zip(records, all_outputs), total=len(records), desc="Writing prompts"):
    save_path = os.path.join(OUT_DIR, rec["out_fname"])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(text + "\n")
