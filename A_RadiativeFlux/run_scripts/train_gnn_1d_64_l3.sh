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
echo "Unified 1D Full Globe training started!"

# Run the unified training script - 1D mode, full dataset
python train_flux.py \
    --model gnn \
    --mode 1d \
    --dataset-type full \
    --dataset /mydata/deepcloud/salman/dataset/h5_data_all_chuncked \
    --save /mydata/deepcloud/yves/results_A_RadiativeFlux/results/gnn_64_l3 \
    --percent 0.1 \
    --subsample 0.1 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --num-cells 81920 \
    --channel-3d 6 \
    --channel-2d 6 \
    --channel-out 4 \
    --height-in 70 \
    --train \
    --test \
    --shuffle \
    --batch-size 2048 \
    --optimizer adamw \
    --clip 1.0 \
    --num-epoch 80 \
    --learning-rate 0.0005 \
    --hidden-dim 64 \
    --layers 3 \
    --dropout 0.0 \
    --edge-channels-in 1 \
    --fully-connected \
    --wandb-mode online

echo "1D Full Globe training completed!" 