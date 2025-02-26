#!/bin/bash

# Set to fail on error
set -e

# Source setup script (optional, in case you want to load a specific conda environment, install packages, setup ssh/gpg/weights-and-biases (wandb) or other keys, ...)
source /mydata/deepcloud/yves/online-datasets/workspace/scripts/setup.sh

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
    --model afno \
    --dataset /mydata/deepcloud/salman/dataset/h5_data_all_chuncked \
    --save /mydata/deepcloud/yves/results_git/afno_clean_GC \
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
    --clip 1.0 \
    --num-epoch 100 \
    --learning-rate 0.0005 \
    --patch-size 1 \
    --hidden-dim 128 \
    --layers 4 \
    --heads 8        \
    --dropout 0.0 \
    --afno-sparsity-threshold 0.01 \
    --wandb-mode online \
