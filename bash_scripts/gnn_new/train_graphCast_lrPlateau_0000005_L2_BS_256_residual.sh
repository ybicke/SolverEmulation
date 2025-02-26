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
python train_column_gnn_scheduler.py \
    --model gnn_graphCast_new_residual \
    --dataset /mydata/deepcloud/salman/dataset/h5_data_all_chuncked \
    --save /mydata/deepcloud/yves/results_git/graphCast_lrPlateau_000001_l2_bs_256_residual \
    --percent 0.1 \
    --subsample 0.1 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --num-cells 81920 \
    --train \
    --test \
    --shuffle \
    --batch-size 256 \
    --optimizer adamw \
    --clip 1 \
    --num-epoch 30 \
    --learning-rate 0.000005 \
    --hidden-dim 128 \
    --dropout 0.2 \
    --layers 2 \
    --lr-schedule-type plateau \
    --wandb-mode online \
