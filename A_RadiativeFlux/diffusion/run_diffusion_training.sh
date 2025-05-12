#!/bin/bash
# Script to train a diffusion model for radiative flux prediction

# Set dataset path - modify as needed
DATASET_PATH="/mydata/deepcloud/salman/dataset/h5_data_all_chuncked"

# Set output path - modify as needed
OUTPUT_PATH="/mydata/deepcloud/yves/results_git/diffusion_model"

# Add necessary paths to PYTHONPATH
export PYTHONPATH=/mydata/deepcloud/yves/SolverEmulation/A_RadiativeFlux

# Change directory to where the train.py file is located
cd /mydata/deepcloud/yves/SolverEmulation/A_RadiativeFlux/diffusion2

# Run the training script
python -m train \
  --dataset ${DATASET_PATH} \
  --save ${OUTPUT_PATH} \
  --percent 0.1 \
  --subsample 0.1 \
  --num-workers 4 \
  --batch-size 32 \
  --learning-rate 0.0001 \
  --max-steps 50000 \
  --height-in 71 \
  --channel-out 4 \
  --channel-3d 6 \
  --channel-2d 6 \
  --cnn-units 64 128 256 512 \
  --cnn-kernel-sizes 2 2 2 2 \
  --dropout 0.1 \
  --num-sampling-steps 25 \
  --deterministic-sampling \
  --sigma-min 0.002 \
  --sigma-max 80.0 \
  --sigma-data 0.5

# Notes:
# - The script uses a small percentage of data (10%) for faster testing
# - Increase --percent and remove --subsample for full training
# - Adjust --num-workers based on your CPU cores available
# - --num-sampling-steps controls sampling quality vs. speed (higher = better quality but slower)
# - Set --deterministic-sampling for deterministic results or remove for stochastic sampling 