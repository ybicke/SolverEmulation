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
    --save /mydata/deepcloud/yves/results_A_RadiativeFlux/results/afno_1d_128_energy_conservation_loss \
    --percent 0.1\
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
    --num-epoch 80 \
    --learning-rate 0.0005 \
    --hidden-dim 128 \
    --fno-blocks 4 \
    --dropout 0.0 \
    --patch-size 1 \
    --afno-sparsity-threshold 0.01 \
    --hard-thresholding-fraction 1.0 \
    --energy-conservation-weight 0.005 \
    --energy-conservation-alpha 0.5 \
    --hidden-size-factor 2 \
    --wandb-mode online

echo "AFNO 1D Full Globe training completed!"
