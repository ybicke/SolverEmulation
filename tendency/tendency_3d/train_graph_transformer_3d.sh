#!/bin/bash

# Set to fail on error
set -e

# Source setup script (optional, in case you want to load a specific conda environment, install packages, setup ssh/gpg/weights-and-biases (wandb) or other keys, ...)
source /mydata/deepcloud/yves/SolverEmulation/A_RadiativeFlux/run_scripts/setup.sh

# Check if GPU available
#nvidia-smi

myproject="deepcloud"
myusername="yves"
myfolder=/mydata/deepcloud/yves/SolverEmulation/tendency/tendency_3d/


# run script

cd $myfolder
echo "Graph Transformer 3D training started!"


# Run the training script with all required parameters
python train_models_3d_tendency.py \
    --model graph_transformer   \
    --dataset-input /mydata/deepcloud/yves/h5_tendency_data_all/inputs \
    --dataset-output /mydata/deepcloud/yves/h5_tendency_data_all/outputs \
    --save /mydata/deepcloud/yves/results-temp/graph_transformer_3d_64_l2\
    --percent 0.1 \
    --subsample 0.1 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --num-cells 81920 \
    --height-in 70 \
    --channel-3d 10 \
    --channel-2d 3 \
    --channel-out 7 \
    --train \
    --test \
    --shuffle \
    --batch-size 1 \
    --optimizer adamw \
    --clip 1 \
    --num-epoch 30 \
    --learning-rate 0.0005 \
    --hidden-dim 64 \
    --layers 2 \
    --dropout 0 \
    --heads 8 \
    --dim-head 8 \
    --mlp-ratio 2.0 \
    --emb-dropout 0 \
    --grid-file-path /mydata/deepcloud/yves/SolverEmulation/data_exploration/icon_grid_0008_R02B05_G.nc \
    --triangle-id 39 \
    --triangle-division-factor 4 \
    --no-fully-connected \
    --no-disable-horizontal \
    --wandb-mode online 