#!/bin/bash

# Set to fail on error
set -e

# Source setup script (optional, in case you want to load a specific conda environment, install packages, setup ssh/gpg/weights-and-biases (wandb) or other keys, ...)
source /mydata/deepcloud/yves/SolverEmulation/A_RadiativeFlux/run_scripts_new/setup.sh

# Check if GPU available
#nvidia-smi

myproject="deepcloud"
myusername="yves"
myfolder=/mydata/deepcloud/yves/SolverEmulation/A_RadiativeFlux/


# run script

cd $myfolder
echo "RNN training started!"


# bash <(sed -n "${line_number}p" run_all_models.sh)


# Run the training script with all required parameters
python train_flux.py \
    --model rnn \
    --mode 1d \
    --dataset-type full \
    --dataset /mydata/deepcloud/salman/dataset/h5_data_all_chuncked \
    --save /mydata/deepcloud/yves/A_RadiativeFlux/results/rnn_medium_hrlu \
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
    --lstm-units 64 64 128 128 256 256 \
    --mlp-units 64 128 \
    --hr-smoothness-weight 0.005 \
    --hr-smoothness-top-levels 30 \
    --lstm-droprate 0.0 \
    --wandb-mode online



