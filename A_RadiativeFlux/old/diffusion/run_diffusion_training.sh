#!/bin/bash

# Set to fail on error
set -e

# Source setup script (optional, in case you want to load a specific conda environment, install packages, setup ssh/gpg/weights-and-biases (wandb) or other keys, ...)
source /mydata/deepcloud/yves/SolverEmulation/A_RadiativeFlux/run_scripts/setup.sh

# Check if GPU available
#nvidia-smi

myproject="deepcloud"
myusername="yves"
myfolder=/mydata/deepcloud/yves/SolverEmulation/A_RadiativeFlux/diffusion


# run script

cd $myfolder
echo "Diffusion training started!"


# Run the training script
python -m train \
  --dataset "/mydata/deepcloud/salman/dataset/h5_data_all_chuncked"\
  --save "/mydata/deepcloud/yves/A_RadiativeFlux/diffusion_model_test7" \
  --percent 0.1 \
  --subsample 0.1 \
  --num-workers 4 \
  --prefetch-factor 2 \
  --batch-size 512 \
  --learning-rate 0.0001 \
  --max-epochs 30 \
  --height-in 71 \
  --channel-out 4 \
  --channel-3d 6 \
  --channel-2d 6 \
  --cnn-units 128 256 512 1024 \
  --cnn-kernel-sizes 1 2 5 7 \
  --dropout 0 \
  --num-sampling-steps 25 \
  --deterministic-sampling \
  --sigma-min 0.002 \
  --sigma-max 80.0 \
  --sigma-data 0.5 \
  --time-embedding-dim 128 \
  --wandb-mode online

# Notes:
# - The script uses a small percentage of data (10%) for faster testing
# - Increase --percent and remove --subsample for full training
# - Adjust --num-workers based on your CPU cores available
# - --num-sampling-steps controls sampling quality vs. speed (higher = better quality but slower)
# - Set --deterministic-sampling for deterministic results or remove for stochastic sampling 