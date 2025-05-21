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
echo "GNN tendency training started!"


# bash <(sed -n "${line_number}p" run_all_models.sh)


# Run the training script with all required parameters
python train_column_tendency_normTarg.py \
    --model gnn_tendency \
    --dataset_input /mydata/deepcloud/yves/h5_tendency_data_all/inputs \
    --dataset_output /mydata/deepcloud/yves/h5_tendency_data_all/outputs \
    --save /mydata/deepcloud/sharyed/results-temp/gnn_32_l2_tendency_normTarg \
    --percent 0.1 \
    --subsample 0.1 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --num-cells 81920 \
    --channel-3d 10 \
    --channel-2d 3 \
    --channels-out 7 \
    --train \
    --test \
    --shuffle \
    --batch-size 2048 \
    --optimizer adamw \
    --clip 1 \
    --num-epoch 60 \
    --learning-rate 0.0005 \
    --hidden-dim 32 \
    --layers 2 \
    --dropout 0.0 \
    --edge-channels-in 1 \
    --fully-connected \
    --wandb-mode online 

