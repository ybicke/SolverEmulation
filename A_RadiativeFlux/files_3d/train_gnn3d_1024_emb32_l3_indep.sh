#!/bin/bash

# Set to fail on error
set -e

# Source setup script (optional, in case you want to load a specific conda environment, install packages, setup ssh/gpg/weights-and-biases (wandb) or other keys, ...)
source /mydata/deepcloud/yves/SolverEmulation/A_RadiativeFlux/run_scripts/setup.sh

# Check if GPU available
#nvidia-smi

myproject="deepcloud"
myusername="yves"
myfolder=/mydata/deepcloud/yves/SolverEmulation/A_RadiativeFlux/


# run script

cd $myfolder
echo "GNN3d training started!"


# bash <(sed -n "${line_number}p" run_all_models.sh)


# Run the training script with all required parameters
python train_models_3d_time.py \
    --model gnn_3d   \
    --dataset /mydata/deepcloud/salman/dataset/h5_data_all_chuncked \
    --save /mydata/deepcloud/yves/A_RadiativeFlux/results/gnn3d_1024_emb32_l3_indep\
    --percent 1 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --num-cells 81920 \
    --height-in 71 \
    --channel-3d 6 \
    --channel-2d 6 \
    --channel-out 4 \
    --train \
    --test \
    --shuffle \
    --batch-size 2 \
    --optimizer adamw \
    --clip 1 \
    --num-epoch 30 \
    --learning-rate 0.0005 \
    --hidden-dim 32 \
    --layers 3 \
    --dropout 0.0 \
    --edge-channels-in 1 \
    --grid-file-path /mydata/deepcloud/yves/SolverEmulation/data_exploration/icon_grid_0008_R02B05_G.nc \
    --triangle-id 39 \
    --triangle-division-factor 4 \
    --fully-connected \
    --disable-horizontal \
    --wandb-mode online