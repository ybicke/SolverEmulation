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
    --model gnn_graphCast_multiMesh_heightFeatures \
    --dataset /mydata/deepcloud/salman/dataset/h5_data_all_chuncked \
    --save /mydata/deepcloud/yves/results_git/graphCast_multiMesh_00001_L3_H70_Emb128_edgeFeatures \
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
    --clip 1 \
    --num-epoch 30 \
    --learning-rate 0.0005 \
    --hidden-dim 128 \
    --dropout 0.0 \
    --layers 3 \
    --max-skip 5 \
    --lr-schedule-type none \
    --edge-channels-in 1 \
    --fully-connected \
    --heights-file "/mydata/deepcloud/yves/SolverEmulation/data_exploration/ml_ecrad_ape_R2B05_myrunscript_ecRad5d_infero5d_70lev_atm_3d_ICONGRID_DOM01_ml_lonlat.nc" \
    --wandb-mode online \
