#!/bin/bash
# Example script to train the CNN model for radiative flux prediction
# Based on the original CnnIg architecture

# Set dataset path - modify as needed
DATASET_PATH="/path/to/your/dataset"

# Set output path - modify as needed
OUTPUT_PATH="./results/cnn_model"

# Train the CNN model
python train_models.py \
  --model cnn \
  --dataset $DATASET_PATH \
  --save $OUTPUT_PATH \
  --batch-size 64 \
  --num-epoch 100 \
  --learning-rate 0.001 \
  --optimizer adamw \
  --channel-out 4 \
  --channel-3d 6 \
  --channel-2d 6 \
  --height-in 70 \
  --hidden-dim 256 \
  --layers 4 \
  --dropout 0.1 \
  --wandb-mode disabled \
  --cnn-units 64 128 256 512 \
  --cnn-kernel-sizes 2 2 2 2 \
  --hr-smoothness-weight 0.05

# Note: You can adjust the parameters based on your specific needs:
# --cnn-units: Controls the size of each CNN layer (number of filters)
# --cnn-kernel-sizes: Controls the max pooling kernel sizes between layers
# --hr-smoothness-weight: Controls smoothness penalty for heating rate profiles
#   (0.0 disables, higher values = smoother profiles) 