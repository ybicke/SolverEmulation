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
echo "Unified 1D with Heating Rate Loss training started!"

# Run the unified training script - 1D mode with heating rate smoothness loss
python train_flux.py \
    --model vit \
    --mode 1d \
    --dataset-type full \
    --dataset /mydata/deepcloud/salman/dataset/h5_data_all_chuncked \
    --save /mydata/deepcloud/yves/A_RadiativeFlux/results/vit_128_hrlu_0005_new \
    --percent 0.1 \
    --subsample 0.1 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --num-cells 81920 \
    --channel-3d 6 \
    --channel-2d 6 \
    --channel-out 4 \
    --height-in 71 \
    --train \
    --test \
    --shuffle \
    --batch-size 2048 \
    --optimizer adamw \
    --clip 1.0 \
    --num-epoch 80 \
    --learning-rate 0.0005 \
    --hidden-dim 128 \
    --mlp-ratio 4 \
    --layers 4 \
    --heads 6 \
    --dim-head 64 \
    --dropout 0.0 \
    --hr-smoothness-weight 0.005 \
    --hr-smoothness-top-levels 30 \
    --wandb-mode online

echo "1D with Heating Rate Loss training completed!" 