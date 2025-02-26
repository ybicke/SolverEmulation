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
python train_column_gnn_new.py \
    --model gnn_graphCast_new \
    --dataset /mydata/deepcloud/salman/dataset/h5_data_all_chuncked \
    --save /mydata/deepcloud/yves/results_git/graphCast_lrConst_l2_new \
    --percent 0.1 \
    --subsample 0.1 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --num-cells 81920 \
    --train \
    --test \
    --shuffle \
    --batch-size 512 \
    --optimizer adamw \
    --clip 32 \
    --num-epoch 30 \
    --learning-rate 0.0005 \
    --hidden-dim 128 \
    --dropout 0.0 \
    --layers 2 \
    --wandb-mode online \
