#!/bin/bash

declare -A tool_prompt_map=(
    ["screw"]="a metal screw"
)

TOOLS_DIR="../../data/images/lora_dataset"
CHECKPOINT_DIR="../../data/lora_finetune/trained_checkpoints"
TRAIN_SCRIPT="../../data/lora_finetune/diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py"

for TOOL in "${!tool_prompt_map[@]}"; do
    echo "🔧 Training for tool: $TOOL"

    # Get the mapped caption
    PROMPT="${tool_prompt_map[$TOOL]} on a table"

    for MODEL in "stabilityai/stable-diffusion-xl-base-1.0"; do
        if [[ "$MODEL" == *"xl-base-1.0" ]]; then
            MODEL_TAG="sdxl"
        else
            MODEL_TAG="sd15"
        fi

        export MODEL_NAME=$MODEL
        export DATASET_NAME="${TOOLS_DIR}/${TOOL}"
        export OUTPUT_DIR="${CHECKPOINT_DIR}/${TOOL}/${MODEL_TAG}"

        echo "🚀 Launching training for $TOOL with $MODEL_TAG"
        accelerate launch --mixed_precision="bf16" $TRAIN_SCRIPT \
            --pretrained_model_name_or_path=${MODEL_NAME} \
            --dataset_name=${DATASET_NAME} \
            --dataloader_num_workers=8 \
            --resolution=1024 \
            --random_flip \
            --train_batch_size=1 \
            --gradient_accumulation_steps=4 \
            --max_train_steps=10000 \
            --learning_rate=1e-04 \
            --max_grad_norm=1 \
            --lr_scheduler="cosine" \
            --lr_warmup_steps=0 \
            --output_dir=${OUTPUT_DIR} \
            --checkpointing_steps=5000 \
            --caption_column="caption" \
            --validation_prompt="$PROMPT" \
            --seed=1337 \
            --resume_from_checkpoint="latest"
    done
done