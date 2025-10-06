#!/bin/bash

# Set to fail on error
set -e

# Source setup script
source /mydata/deepcloud/yves/SolverEmulation/A_RadiativeFlux/run_scripts_new/setup.sh

# Check if GPU available
nvidia-smi

myproject="deepcloud"
myusername="yves"
myfolder=/mydata/deepcloud/yves/SolverEmulation/A_RadiativeFlux/

# Run script
cd $myfolder
echo "Unified 3D Triangle training started!"

# Run the unified training script - 3D mode, triangle dataset
python unified_train_flux.py \
    --model gnn_3d \
    --mode 3d \
    --dataset-type triangle \
    --dataset /mydata/deepcloud/salman/dataset/h5_data_all_chuncked \
    --save /mydata/deepcloud/yves/A_RadiativeFlux/results-new/gnn3d_triangle \
    --percent 0.1 \
    --subsample 0.1 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --num-cells 81920 \
    --channel-3d 6 \
    --channel-2d 6 \
    --channel-out 4 \
    --height-in 70 \
    --triangle-id 1 \
    --triangle-division-factor 1 \
    --grid-file-path /mydata/deepcloud/yves/SolverEmulation/data_exploration/icon_grid_0008_R02B05_G.nc \
    --train \
    --test \
    --shuffle \
    --batch-size 2 \
    --optimizer adamw \
    --clip 1.0 \
    --num-epoch 30 \
    --learning-rate 0.0005 \
    --hidden-dim 32 \
    --layers 3 \
    --dropout 0.0 \
    --edge-channels-in 1 \
    --no-fully-connected \
    --no-disable-horizontal \
    --max-hops 1 \
    --wandb-mode online

echo "3D Triangle training completed!" 