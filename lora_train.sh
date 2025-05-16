export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export DATASET_NAME="../../data/lora_finetune/datasets/screwdriver"
export OUTPUT_DIR="../../data/lora_finetune/trained_checkpoints/screwdriver"
export CHECKPOINT_PATH="../../data/lora_finetune/trained_checkpoints/screwdriver/sdxl/checkpoint-5000"

accelerate launch --mixed_precision="bf16" ../../data/lora_finetune/diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py \
    --pretrained_model_name_or_path=${MODEL_NAME} \
    --dataset_name=${DATASET_NAME} \
    --dataloader_num_workers=8 \
    --caption_column="text" \
    --resolution=1024 \
    --random_flip \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=11000 \
    --learning_rate=1e-04 \
    --max_grad_norm=1 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=0 \
    --output_dir=${OUTPUT_DIR} \
    --checkpointing_steps=5000 \
    --caption_column="caption" \
    --validation_prompt="a screwdriver on a table" \
    --seed=1337 \
    --resume_from_checkpoint=${CHECKPOINT_PATH}