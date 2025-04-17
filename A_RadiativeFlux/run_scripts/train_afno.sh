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
echo training$line_number started!


# bash <(sed -n "${line_number}p" run_all_models.sh)


# Run the training script with all required parameters
python train_models.py \
    --model afno \
    --dataset /mydata/deepcloud/salman/dataset/h5_data_all_chuncked \
    --save /mydata/deepcloud/yves/Results-A_RadiativeFlux/afno_test1 \
    --percent 0.1 \
    --subsample 0.1 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --num-cells 81920 \
    --channel-3d 6 \
    --channel-2d 6 \
    --channel-out 4 \
    --height-in 71 \
    --train \
    --test \
    --shuffle \
    --batch-size 2048 \
    --optimizer adamw \
    --clip 1.0 \
    --num-epoch 30 \
    --learning-rate 0.0005 \
    --patch-size 1 \
    --hidden-dim 128 \
    --layers 4 \
    --dropout 0.0 \
    --scale-output \
    --fno-blocks 8 \
    --afno-sparsity-threshold 0.01 \
    --hard-thresholding-fraction 1.0 \
    --hidden-size-factor 1 \
    --double-skip \
    --mlp-ratio 4.0 \
    --wandb-mode online \
