#!/bin/bash

# Set to fail on error
set -e

# Source setup script
source /mydata/deepcloud/yves/SolverEmulation/A_RadiativeFlux/run_scripts_new/setup.sh

# Check if GPU available
nvidia-smi

myproject="deepcloud"
myusername="yves"
myfolder=/mydata/deepcloud/yves/SolverEmulation/A_RadiativeFlux/

# Run script
cd $myfolder
echo "AFNO 1D Full Globe training (30% data - 44M samples) started!"



python train_flux.py \
    --model afno \
    --mode 1d \
    --dataset-type full \
    --dataset /mydata/deepcloud/salman/dataset/h5_data_all_chuncked \
    --save /mydata/deepcloud/yves/A_RadiativeFlux/results-new/afno_1d_64_full_1percent_sparse_051 \
    --percent 0.1 \
    --subsample 0.1 \
    --num-workers 8 \
    --prefetch-factor 4 \
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
    --num-epoch 60 \
    --learning-rate 0.0005 \
    --hidden-dim 64 \
    --layers 6 \
    --dropout 0.0 \
    --patch-size 1 \
    --mlp-ratio 2 \
    --fno-blocks 4 \
    --afno-sparsity-threshold 0.51 \
    --hard-thresholding-fraction 1.0 \
    --hidden-size-factor 1 \
    --wandb-mode online

echo "AFNO 1D Full Globe training completed!"
