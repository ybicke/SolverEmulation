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
echo "ViT 1D Full Globe training started!"

# Run the training script with ViT model configuration
python train_flux.py \
    --model vit \
    --mode 1d \
    --dataset-type full \
    --dataset /mydata/deepcloud/salman/dataset/h5_data_all_chuncked \
    --save /mydata/deepcloud/yves/A_RadiativeFlux/results/vit_1d_full_30percent_256_l8_h8 \
    --memory-efficient-test \
    --test-chunk-size 500 \
    --percent 0.5 \
    --subsample 0.6 \
    --num-workers 8 \
    --prefetch-factor 4 \
    --num-cells 81920 \
    --channel-3d 6 \
    --channel-2d 6 \
    --channel-out 4 \
    --height-in 71 \
    --no-train \
    --test \
    --shuffle \
    --batch-size 2048 \
    --optimizer adamw \
    --clip 1.0 \
    --num-epoch 30 \
    --learning-rate 0.0007 \
    --hidden-dim 256 \
    --layers 8 \
    --heads 8 \
    --dropout 0.0 \
    --emb-dropout 0.0 \
    --patch-size 1 \
    --dim-head 32 \
    --mlp-ratio 2 \
    --wandb-mode online

echo "ViT 1D Full Globe training completed!" 