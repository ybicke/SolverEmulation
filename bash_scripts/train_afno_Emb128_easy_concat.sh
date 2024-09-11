#!/bin/bash

# Set to fail on error
set -e

# Source setup script (optional, in case you want to load a specific conda environment, install packages, setup ssh/gpg/weights-and-biases (wandb) or other keys, ...)
source /mydata/deepcloud/yves/online-datasets/workspace/scripts/setup.sh

# Check if GPU available
#nvidia-smi

myproject="deepcloud"
myusername="yves"
myfolder=/mydata/deepcloud/yves/online-datasets/workspace/


# run script

cd $myfolder
echo training$line_number started!


# bash <(sed -n "${line_number}p" run_all_models.sh)


# Run the training script with specified parameters
python train_column_concat_easy.py \
    --model afno \
    --dataset /mydata/deepcloud/salman/dataset/h5_data_all_chuncked \
    --save /mydata/deepcloud/yves/online-datasets/workspace/results/afno_column_1percent_8_easy_concat2\
    --percent 0.1 \
    --subsample 0.1 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --num-cells 81920 \
    --train \
    --test \
    --shuffle \
    --batch-size 2048 \
    --vbatch 1 \
    --optimizer adamw \
    --clip 1.0 \
    --num-epoch 150 \
    --learning-rate 0.0005 \
    --patch-size 1 \
    --vit-hidden-dim 128 \
    --vit-layers 4 \
    --vit-heads 6 \
    --vit-dropout 0.0 \
    --wandb-mode online \
