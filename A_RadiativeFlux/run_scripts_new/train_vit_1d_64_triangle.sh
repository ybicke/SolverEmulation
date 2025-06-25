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
echo "Unified 1D Triangle training started!"

# Run the unified training script - 1D mode, triangle dataset
python train_flux.py \
    --model vit \
    --mode 1d \
    --dataset-type triangle \
    --dataset /mydata/deepcloud/salman/dataset/h5_data_all_chuncked \
    --save /mydata/deepcloud/yves/A_RadiativeFlux/results/vit_1d_triangle_64 \
    --percent 1.0 \
    --subsample 1.0 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --num-cells 81920 \
    --channel-3d 6 \
    --channel-2d 6 \
    --channel-out 4 \
    --height-in 70 \
    --triangle-id 1 \
    --triangle-division-factor 4 \
    --train \
    --test \
    --shuffle \
    --batch-size 1024 \
    --optimizer adamw \
    --clip 1.0 \
    --num-epoch 100 \
    --learning-rate 0.0005 \
    --hidden-dim 64 \
    --heads 8 \
    --dim-head 8 \
    --mlp-ratio 4.0 \
    --layers 4 \
    --dropout 0.0 \
    --edge-channels-in 1 \
    --fully-connected \
    --wandb-mode online

echo "1D Triangle training completed!" 