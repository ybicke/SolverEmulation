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
echo "GNN 3D Tendency training started!"

# Run the training script with all required parameters
python train_tendency.py \
    --model gnn_3d \
    --mode 3d \
    --dataset-type triangle \
    --dataset-input /mydata/deepcloud/yves/h5_tendency_data_all/inputs \
    --dataset-output /mydata/deepcloud/yves/h5_tendency_data_all/outputs \
    --input-stats-file /mydata/deepcloud/shared/h5_tendency_all/normalizer_stats_per_feat_updated.pickle \
    --target-stats-file /mydata/deepcloud/shared/h5_tendency_all/normalizer_stats_per_feat_y2_no_temp.pickle \
    --save /mydata/deepcloud/yves/results-new/gnn_3d_tendency_128_l2_100 \
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
    --hidden-dim 128 \
    --layers 2 \
    --grid-file-path /mydata/deepcloud/yves/SolverEmulation/data_exploration/icon_grid_0008_R02B05_G.nc \
    --triangle-id 39 \
    --triangle-division-factor 4 \
    --fully-connected \
    --no-disable-horizontal \
    --edge-channels-in 1 \
    --wandb-mode online 