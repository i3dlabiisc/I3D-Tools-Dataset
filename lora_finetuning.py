# import os
# import json
# import torch
# from PIL import Image
# from torch.utils.data import Dataset, DataLoader
# from peft import get_peft_model, LoraConfig, TaskType
# import open_clip
# from tqdm import tqdm

# # === SETUP ===
# PROMPT_DIR = "/home/nahar/Desktop/diffusion/hand_tools_dataset/data/images/composited/prompt02"
# IMAGE_DIR = "/home/nahar/Desktop/diffusion/hand_tools_dataset/data/images/refined/prompt02"
# KNOWN_TOOLS = ["gear", "nut", "hammer", "screwdriver", "washer", "measuring tape", "plier", "saw", "scissors", "utility knife", "wrench","ball bearing"," o ring","screw","spring"]
# label2id = {tool: i for i, tool in enumerate(KNOWN_TOOLS)}

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
# tokenizer = open_clip.get_tokenizer("ViT-B-32")
# model.to(device)

# # Freeze all except LoRA
# for param in model.parameters():
#     param.requires_grad = False

# peft_config = LoraConfig(
#     r=8,
#     lora_alpha=16,
#     target_modules=[
#         "visual.transformer.resblocks.0.attn.out_proj",
#         "visual.transformer.resblocks.0.mlp.c_fc",
#         "visual.transformer.resblocks.0.mlp.c_proj",
#         "visual.transformer.resblocks.1.attn.out_proj",
#         "visual.transformer.resblocks.1.mlp.c_fc",
#         "visual.transformer.resblocks.1.mlp.c_proj",
#         "visual.transformer.resblocks.2.attn.out_proj",
#         "visual.transformer.resblocks.2.mlp.c_fc",
#         "visual.transformer.resblocks.2.mlp.c_proj"
#     ],
#     lora_dropout=0.1,
#     bias="none",
#     task_type=TaskType.FEATURE_EXTRACTION
# )
# model = get_peft_model(model, peft_config)

# # === UTILS ===
# def extract_tools_from_prompt(prompt_text):
#     prompt_text = prompt_text.lower()
#     return [tool for tool in KNOWN_TOOLS if f" {tool}" in prompt_text or f"{tool}," in prompt_text]

# # === DYNAMIC DATASET CREATION ===
# class MultiToolDataset(Dataset):
#     def __init__(self, prompt_dir, image_dir, transform):
#         self.items = []
#         self.transform = transform

#         for fname in os.listdir(prompt_dir):
#             if not fname.endswith(".txt"):
#                 continue
#             base_name = fname.replace("_prompt.txt", "")
#             txt_path = os.path.join(prompt_dir, fname)

#             with open(txt_path, "r") as f:
#                 prompt = f.read().strip()

#             labels = extract_tools_from_prompt(prompt)
#             if not labels:
#                 continue

#             # Match both _base.jpg and _v4.jpg
#             for suffix in ["_base.jpg", "_v4.jpg"]:
#                 img_path = os.path.join(image_dir, base_name + suffix)
#                 if os.path.exists(img_path):
#                     self.items.append({
#                         "img_path": img_path,
#                         "tools": labels
#                     })

#     def __len__(self):
#         return len(self.items)

#     def __getitem__(self, idx):
#         data = self.items[idx]
#         img = self.transform(Image.open(data["img_path"]).convert("RGB"))
#         label = torch.zeros(len(KNOWN_TOOLS))
#         for tool in data["tools"]:
#             label[label2id[tool]] = 1.0
#         return {"image": img, "label": label}

# # === DATASET + DATALOADER ===
# dataset = MultiToolDataset(PROMPT_DIR, IMAGE_DIR, preprocess)
# loader = DataLoader(dataset, batch_size=16, shuffle=True)

# # === TRAINING ===
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
# loss_fn = torch.nn.BCEWithLogitsLoss()

# for epoch in range(5):
#     model.train()
#     total_loss = 0
#     for batch in tqdm(loader, desc=f"Epoch {epoch}"):
#         images = batch["image"].to(device)
#         labels = batch["label"].to(device)

#         with torch.no_grad():
#             img_feats = model.encode_image(images)

#         if not hasattr(model, 'classifier'):
#             model.classifier = torch.nn.Linear(model.visual.output_dim, len(label2id)).to(device)

#         logits = model.classifier(img_feats)

#         loss_fn = torch.nn.BCEWithLogitsLoss()
#         loss = loss_fn(logits, labels)

#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         total_loss += loss.item()

#     print(f"Epoch {epoch} - Avg Loss: {total_loss/len(loader):.4f}")

# # === SAVE FINETUNED MODEL ===
# os.makedirs("lora_clip_finetuned", exist_ok=True)
# model.save_pretrained("lora_clip_finetuned")


import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from peft import get_peft_model, LoraConfig, TaskType
import open_clip
from tqdm import tqdm

# === SETUP ===
PROMPT_DIR = "/home/nahar/Desktop/diffusion/hand_tools_dataset/data/images/composited/prompt02"
IMAGE_DIR = "/home/nahar/Desktop/diffusion/hand_tools_dataset/data/images/refined/prompt02"
KNOWN_TOOLS = ["gear", "nut", "hammer", "screwdriver", "washer", "measuring tape", "plier", "saw", "scissors", "utility knife", "wrench", "ball bearing", "o ring", "screw", "spring"]
label2id = {tool: i for i, tool in enumerate(KNOWN_TOOLS)}

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model.to(device)

# Freeze all except LoRA
for param in model.parameters():
    param.requires_grad = False

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        f"visual.transformer.resblocks.{i}.{layer}"
        for i in range(10)  # adjust if needed
        for layer in ["attn.out_proj", "mlp.c_fc", "mlp.c_proj"]
    ],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION
)
model = get_peft_model(model, peft_config)

# Add classifier head for multi-label
model.classifier = torch.nn.Linear(model.visual.output_dim, len(label2id)).to(device)

# === UTILS ===
def extract_tools_from_prompt(prompt_text):
    prompt_text = prompt_text.lower()
    return [tool for tool in KNOWN_TOOLS if f" {tool}" in prompt_text or f"{tool}," in prompt_text]

# === DATASET ===
class MultiToolDataset(Dataset):
    def __init__(self, prompt_dir, image_dir, transform):
        self.items = []
        self.transform = transform

        for fname in os.listdir(prompt_dir):
            if not fname.endswith(".txt"):
                continue
            base_name = fname.replace("_prompt.txt", "")
            txt_path = os.path.join(prompt_dir, fname)

            with open(txt_path, "r") as f:
                prompt = f.read().strip()

            labels = extract_tools_from_prompt(prompt)
            if not labels:
                continue

            for suffix in ["_base.jpg", "_v4.jpg"]:
                img_path = os.path.join(image_dir, base_name + suffix)
                if os.path.exists(img_path):
                    self.items.append({
                        "img_path": img_path,
                        "tools": labels
                    })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        data = self.items[idx]
        img = self.transform(Image.open(data["img_path"]).convert("RGB"))
        label = torch.zeros(len(KNOWN_TOOLS))
        for tool in data["tools"]:
            label[label2id[tool]] = 1.0
        return {"image": img, "label": label}

# === LOAD DATA ===
dataset = MultiToolDataset(PROMPT_DIR, IMAGE_DIR, preprocess)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# === TRAINING ===
optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(model.classifier.parameters()), lr=1e-4
)
loss_fn = torch.nn.BCEWithLogitsLoss()

for epoch in range(50):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc=f"Epoch {epoch}"):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        with torch.no_grad():
            img_feats = model.encode_image(images)

        logits = model.classifier(img_feats)
        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    print(f"✅ Epoch {epoch} - Avg Loss: {total_loss/len(loader):.4f}")

# === SAVE MODEL ===
os.makedirs("lora_clip_finetuned", exist_ok=True)
model.save_pretrained("lora_clip_finetuned")
