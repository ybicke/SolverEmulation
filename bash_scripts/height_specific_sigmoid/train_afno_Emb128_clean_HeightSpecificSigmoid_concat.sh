#!/bin/bash

# Set to fail on error
set -e

# Source setup script (optional, in case you want to load a specific conda environment, install packages, setup ssh/gpg/weights-and-biases (wandb) or other keys, ...)
source /mydata/deepcloud/yves/online-datasets/workspace/scripts/setup.sh

# Check if GPU available
#nvidia-smi

myproject="deepcloud"
myusername="yves"
myfolder=/mydata/deepcloud/shared/SolverEmulation


# run script

cd $myfolder
echo training$line_number started!


# bash <(sed -n "${line_number}p" run_all_models.sh)


# Run the training script with specified parameters
python train_column.py \
    --model afno_clean_heightDepSigmoid_concat \
    --dataset /mydata/deepcloud/salman/dataset/h5_data_all_chuncked \
    --save /mydata/deepcloud/shared/results-temp/afno_column_1percent_Emb128_HeightSpecificSigmoid_concat \
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
    --num-epoch 100 \
    --learning-rate 0.0005 \
    --patch-size 1 \
    --vit-hidden-dim 128 \
    --vit-layers 4 \
    --vit-heads 8        \
    --vit-dropout 0.0 \
    --afno-sparsity-threshold 0.01 \
    --gaussian_params_file_LWDown /mydata/deepcloud/yves/results_git/data_histograms/histogram_fit_lwDown/fitted_gaussians_lwDown.npz \
    --gaussian_params_file_LWUp /mydata/deepcloud/yves/results_git/data_histograms/histogram_fit_lwUp/fitted_gaussians_lwUp.npz \
    --wandb-mode online \
