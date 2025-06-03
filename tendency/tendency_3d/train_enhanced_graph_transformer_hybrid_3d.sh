#!/bin/bash

# Set to fail on error
set -e

# Source setup script
source /mydata/deepcloud/yves/SolverEmulation/A_RadiativeFlux/run_scripts/setup.sh

myproject="deepcloud"
myusername="yves"
myfolder=/mydata/deepcloud/yves/SolverEmulation/tendency/tendency_3d/

# Run script
cd $myfolder
echo "Enhanced Hybrid Graph Transformer 3D training started!"

# Run the training script with enhanced parameters
python train_models_3d_tendency.py \
    --model enhanced_graph_transformer_hybrid_3d   \
    --dataset-input /mydata/deepcloud/yves/h5_tendency_data_all/inputs \
    --dataset-output /mydata/deepcloud/yves/h5_tendency_data_all/outputs \
    --save /mydata/deepcloud/yves/results-temp/enhanced_graph_transformer_hybrid_3d_64_l4_k2_drop03\
    --percent 1 \
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
    --learning-rate 0.0003 \
    --hidden-dim 64 \
    --layers 4 \
    --dropout 0.3 \
    --heads 8 \
    --dim-head 8 \
    --mlp-ratio 3.0 \
    --grid-file-path /mydata/deepcloud/yves/SolverEmulation/data_exploration/icon_grid_0008_R02B05_G.nc \
    --triangle-id 39 \
    --triangle-division-factor 4 \
    --no-fully-connected \
    --no-disable-horizontal \
    --max-hops 2 \
    --wandb-mode online 