#!/bin/bash

# Set to fail on error
set -e

# Source setup script (optional, in case you want to load a specific conda environment, install packages, setup ssh/gpg/weights-and-biases (wandb) or other keys, ...)
source /mydata/deepcloud/yves/SolverEmulation/bash_scripts/setup.sh

# Check if GPU available
#nvidia-smi

myproject="deepcloud"
myusername="yves"
myfolder=/mydata/deepcloud/yves/SolverEmulation


# run script

cd $myfolder
echo training$line_number started!


# bash <(sed -n "${line_number}p" run_all_models.sh)


# Run the training script with specified parameters
python train_column_gnn.py \
    --model gnn_graphCast_column \
    --dataset /mydata/deepcloud/salman/dataset/h5_data_all_chuncked \
    --save /mydata/deepcloud/yves/results_git/gnn_graphCast_Emb512_const_L8_tiny \
    --percent 0.001 \
    --subsample 0.001 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --num-cells 81920 \
    --train \
    --test \
    --shuffle \
    --batch-size 32 \
    --optimizer adamw \
    --clip 1 \
    --num-epoch 100 \
    --learning-rate 0.0001 \
    --hidden-dim 512 \
    --dropout 0.0 \
    --layers 8 \
    --lr-schedule-type none \
    --wandb-mode online \
