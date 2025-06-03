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
echo "GNN tendency triangle training started!"

# Run the training script with all required parameters
python train_column_tendency_1d_tringle.py \
    --model gnn \
    --dataset-input /mydata/deepcloud/yves/h5_tendency_data_all/inputs \
    --dataset-output /mydata/deepcloud/yves/h5_tendency_data_all/outputs \
    --save /mydata/deepcloud/yves/results-temp/gnn_64_l2_tendency_1d_triangle_fully_connected \
    --percent 1 \
    --subsample 1 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --num-cells 81920 \
    --channel-3d 10 \
    --channel-2d 3 \
    --channels-out 7 \
    --train \
    --test \
    --shuffle \
    --batch-size 1024 \
    --optimizer adamw \
    --clip 1 \
    --num-epoch 100 \
    --learning-rate 0.0005 \
    --hidden-dim 64 \
    --layers 2 \
    --dropout 0.0 \
    --triangle-id 39 \
    --triangle-division-factor 4 \
    --fully-connected \
    --edge-channels-in 1 \
    --wandb-mode online 