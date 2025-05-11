#!/bin/bash
# Example script to train the diffusion model for radiative flux prediction

# Set dataset path - modify as needed
DATASET_PATH="/path/to/your/dataset"

# Set output path - modify as needed
OUTPUT_PATH="./results/diffusion_model"

# Train the diffusion model
python train_models.py \
  --model diffusion \
  --dataset $DATASET_PATH \
  --save $OUTPUT_PATH \
  --batch-size 32 \
  --num-epoch 100 \
  --learning-rate 0.0001 \
  --optimizer adamw \
  --channel-out 4 \
  --channel-3d 6 \
  --channel-2d 6 \
  --height-in 71 \
  --hidden-dim 256 \
  --layers 4 \
  --dropout 0.1 \
  --wandb-mode disabled \
  --cnn-units 64 128 256 512 \
  --cnn-kernel-sizes 2 2 2 2 \
  --num-sampling-steps 25 \
  --deterministic-sampling \
  --sigma-min 0.002 \
  --sigma-max 80.0 \
  --sigma-data 0.5 \
  --hr-smoothness-weight 0.0

# Notes on diffusion model parameters:
# --num-sampling-steps: Number of steps for the diffusion sampling process.
#    Higher values (e.g., 100-1000) give better quality but slower sampling.
#    Lower values (e.g., 10-50) are faster but may reduce quality.
#
# --deterministic-sampling: Flag to use deterministic sampling.
#    If omitted, stochastic sampling will be used (better for more steps).
#
# --sigma-min, --sigma-max, --sigma-data: Control the noise schedule.
#    Default values often work well but can be tuned for your specific data.