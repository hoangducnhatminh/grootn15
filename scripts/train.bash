#!/bin/bash

# Within IsaacLabEvalTasks directory
cd submodules/Isaac-GR00T

# Check required environment variables
if [ -z "$DATASET_PATH" ]; then
    echo "Error: DATASET_PATH is not set."
    echo "Example: export DATASET_PATH=/home/minh/Workspace/projects/IsaacLabEvalTasks/dataset/generated_dataset_annotated_gr1/lerobot"
    exit 1
fi

# Set default OUTPUT_DIR if not provided
if [ -z "$OUTPUT_DIR" ]; then
    export OUTPUT_DIR="/home/minh/Workspace/projects/IsaacLabEvalTasks/dataset/generated_dataset_annotated_gr1/gr00t_$(date +%Y%m%d_%H%M%S)"
    echo "Warning: OUTPUT_DIR not set. Using default: $OUTPUT_DIR"
fi

echo "Starting GR00T fine-tuning..."
echo "Dataset Path: $DATASET_PATH"
echo "Output Directory: $OUTPUT_DIR"

source .venv/bin/activate
# Nut pouring / Exhaust pipe sorting fine-tuning
python scripts/gr00t_finetune_custom.py \
    --dataset-path="${DATASET_PATH}" \
    --output-dir="${OUTPUT_DIR}" \
    --base-model-path="nvidia/GR00T-N1-2B" \
    --data-config="gr1_arms_only" \
    --embodiment-tag="gr1" \
    --batch-size=96 \
    --max-steps=20000 \
    --num-gpus=8 \
    --save-steps=5000 \
    --no-tune-llm  \
    --tune-visual \
    --tune-projector \
    --tune-diffusion-model \
    --no-resume \
    --dataloader-num-workers=16 \
    --report-to="wandb"