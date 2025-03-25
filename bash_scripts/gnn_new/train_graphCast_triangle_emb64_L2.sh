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
python train_column_triangle.py \
    --model gnn_graphCast_triangle \
    --dataset /mydata/deepcloud/salman/dataset/h5_data_all_chuncked \
    --save /mydata/deepcloud/yves/results_git/gnn_graphCast_triangle_emb64_L2 \
    --percent 0.1 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --num-cells 81920 \
    --train \
    --test \
    --shuffle \
    --batch-size 1\
    --optimizer adamw \
    --clip 1 \
    --num-epoch 30 \
    --learning-rate 0.0005 \
    --hidden-dim 64 \
    --dropout 0.0 \
    --layers 2 \
    --triangle-id 9 \
    --wandb-mode online \
