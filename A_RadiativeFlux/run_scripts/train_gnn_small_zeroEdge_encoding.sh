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
echo "GNN training started!"


# bash <(sed -n "${line_number}p" run_all_models.sh)


# Run the training script with all required parameters
python train_models.py \
    --model gnn_zeroEdge_encoding \
    --dataset /mydata/deepcloud/salman/dataset/h5_data_all_chuncked \
    --save /mydata/deepcloud/yves/A_RadiativeFlux/results/gnn_small_zeroEdge_encoding \
    --percent 0.1 \
    --subsample 0.1 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --num-cells 81920 \
    --channel-3d 6 \
    --channel-2d 6 \
    --channel-out 4 \
    --train \
    --test \
    --shuffle \
    --batch-size 512 \
    --optimizer adamw \
    --clip 1 \
    --num-epoch 30 \
    --learning-rate 0.0005 \
    --hidden-dim 32 \
    --layers 3 \
    --dropout 0.0 \
    --edge-channels-in 1 \
    --fully-connected \
    --wandb-mode online

