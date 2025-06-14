#!/bin/bash

# Set to fail on error
set -e

# Source setup script (optional, in case you want to load a specific conda environment, install packages, setup ssh/gpg/weights-and-biases (wandb) or other keys, ...)
source /mydata/deepcloud/yves/SolverEmulation/B_Tendency/run_scripts/setup.sh

# Check if GPU available
#nvidia-smi

myproject="deepcloud"
myusername="yves"
myfolder=/mydata/deepcloud/yves/SolverEmulation/B_Tendency

# run script

cd $myfolder
echo "GNN tendency triangle training started!"

# Run the training script with all required parameters
python train_tendency.py \
    --model gnn \
    --mode 1d \
    --dataset-type triangle \
    --dataset-input /mydata/deepcloud/yves/h5_tendency_data_all/inputs \
    --dataset-output /mydata/deepcloud/yves/h5_tendency_data_all/outputs \
    --input-stats-file /mydata/deepcloud/shared/h5_tendency_all/normalizer_stats_per_feat_updated.pickle \
    --target-stats-file /mydata/deepcloud/shared/h5_tendency_all/normalizer_stats_per_feat_y2_no_temp.pickle \
    --save /mydata/deepcloud/yves/results-temp/gnn_64_l2_tendency_1d_triangle_fully_connected \
    --percent 1 \
    --subsample 1 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --num-cells 81920 \
    --channel-3d 10 \
    --channel-2d 3 \
    --channels-out 7 \
    --no-train \
    --test \
    --shuffle \
    --batch-size 1024 \
    --optimizer adamw \
    --clip 1 \
    --num-epoch 100 \
    --learning-rate 0.0005 \
    --hidden-dim 64 \
    --layers 2 \
    --dropout 0.0 \
    --triangle-id 39 \
    --triangle-division-factor 4 \
    --fully-connected \
    --edge-channels-in 1 \
    --wandb-mode online 