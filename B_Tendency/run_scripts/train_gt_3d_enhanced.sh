#!/bin/bash

# Set to fail on error
set -e

# Source setup script
source /mydata/deepcloud/yves/SolverEmulation/B_Tendency/run_scripts/setup.sh

# Check if GPU available
nvidia-smi

myproject="deepcloud"
myusername="yves"
myfolder=/mydata/deepcloud/yves/SolverEmulation/B_Tendency/

# run script
cd $myfolder
echo "Graph Transformer 3D Enhanced training started!"

# Run the training script with all required parameters
python train_tendency.py \
    --model gt_enhanced \
    --mode 3d \
    --dataset-type triangle \
    --dataset-input /mydata/deepcloud/yves/h5_tendency_data_all/inputs \
    --dataset-output /mydata/deepcloud/yves/h5_tendency_data_all/outputs \
    --save /mydata/deepcloud/yves/results-new/gt_enhanced_64_l4_drop03_triangle39_k2 \
    --percent 1.0 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --num-cells 81920 \
    --height 70 \
    --channel-3d 10 \
    --channel-2d 3 \
    --channels-out 7 \
    --train \
    --test \
    --shuffle \
    --batch-size 1 \
    --optimizer adamw \
    --clip 1.0 \
    --num-epoch 100 \
    --learning-rate 0.0005 \
    --hidden-dim 64 \
    --layers 4 \
    --dropout 0.3 \
    --heads 8 \
    --dim-head 8 \
    --mlp-ratio 2.0 \
    --emb-dropout 0.3 \
    --grid-file-path /mydata/deepcloud/yves/SolverEmulation/data_exploration/icon_grid_0008_R02B05_G.nc \
    --triangle-id 39 \
    --triangle-division-factor 4 \
    --no-fully-connected \
    --no-disable-horizontal \
    --max-hops 2 \
    --wandb-mode online 