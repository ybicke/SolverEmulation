#!/bin/bash

# Set to fail on error
set -e

# Source setup script (optional, in case you want to load a specific conda environment, install packages, setup ssh/gpg/weights-and-biases (wandb) or other keys, ...)
source /mydata/deepcloud/yves/SolverEmulation/A_RadiativeFlux/run_scripts/setup.sh

# Check if GPU available
#nvidia-smi

myproject="deepcloud"
myusername="yves"
myfolder=/mydata/deepcloud/yves/SolverEmulation/tendency/tendency

# run script

cd $myfolder
echo "AFNO tendency triangle training started!"

# Run the training script with all required parameters
python train_models_1d_tendency_triangle.py \
    --model afno_tendency \
    --dataset-input /mydata/deepcloud/yves/h5_tendency_data_all/inputs \
    --dataset-output /mydata/deepcloud/yves/h5_tendency_data_all/outputs \
    --save /mydata/deepcloud/yves/results-temp/afno_128_l4_tendency_1d_triangle \
    --percent 1 \
    --subsample 1 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --num-cells 81920 \
    --channel-3d 10 \
    --channel-2d 3 \
    --channels-out 7 \
    --patch-size 1 \
    --height 70 \
    --train \
    --test \
    --shuffle \
    --batch-size 1024 \
    --optimizer adamw \
    --clip 1 \
    --num-epoch 100 \
    --learning-rate 0.0005 \
    --hidden-dim 128 \
    --layers 4 \
    --dropout 0.0 \
    --emb-dropout 0.0 \
    --mlp-ratio 4.0 \
    --fno-blocks 8 \
    --hidden-size-factor 1 \
    --hard-thresholding-fraction 1 \
    --afno-sparsity-threshold 0.01 \
    --triangle-id 39 \
    --triangle-division-factor 4 \
    --wandb-mode online 