#!/bin/bash

# Set to fail on error
set -e

# Source setup script (optional, in case you want to load a specific conda environment, install packages, setup ssh/gpg/weights-and-biases (wandb) or other keys, ...)
source /mydata/deepcloud/yves/SolverEmulation/A_RadiativeFlux/run_scripts/setup.sh

# Check if GPU available
#nvidia-smi

myproject="deepcloud"
myusername="yves"
myfolder=/mydata/deepcloud/yves/SolverEmulation/A_RadiativeFlux/


# run script

cd $myfolder
echo "UNET training started!"


# bash <(sed -n "${line_number}p" run_all_models.sh)


# Train the CNN model
python train_models_time.py \
  --model unet \
  --dataset /mydata/deepcloud/salman/dataset/h5_data_all_chuncked \
  --save /mydata/deepcloud/yves/A_RadiativeFlux/results/unet_test \
  --percent 0.1 \
  --subsample 0.1 \
  --num-workers 4 \
  --prefetch-factor 2 \
  --train \
  --test \
  --shuffle \
  --batch-size 512 \
  --num-epoch 30 \
  --learning-rate 0.0005 \
  --optimizer adamw \
  --channel-out 4 \
  --channel-3d 6 \
  --channel-2d 6 \
  --height-in 71 \
  --layers 4 \
  --dropout 0 \
  --cnn-units 128 256 512 1024 \
  --cnn-kernel-sizes 1 2 5 7 \
  --hr-smoothness-weight 0.05\
  --wandb-mode online \


# Note: You can adjust the parameters based on your specific needs:
# --cnn-units: Controls the size of each CNN layer (number of filters)
# --cnn-kernel-sizes: Controls the max pooling kernel sizes between layers
# --hr-smoothness-weight: Controls smoothness penalty for heating rate profiles
#   (0.0 disables, higher values = smoother profiles) 