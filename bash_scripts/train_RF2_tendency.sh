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
python train_column_tendency_RF.py \
    --model rf \
    --dataset_input /mydata/deepcloud/yves/h5_tendency_data_all/inputs \
    --dataset_output /mydata/deepcloud/yves/h5_tendency_data_all/outputs \
    --save /mydata/deepcloud/shared/results-temp/afno_column_1percent_Emb128_clean_tendency_RF_2 \
    --percent 0.1 \
    --subsample 0.1 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --train \
    --test \
    --shuffle \
    --batch-size 2048 \
    --wandb-mode online \
    --n-estimators 40 \
    --max-depth 20 \
