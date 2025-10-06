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
echo "RNN 1D Full Globe training started!"

# Run the training script with RNN model configuration
python train_flux.py \
    --model rnn \
    --mode 1d \
    --dataset-type full \
    --dataset /mydata/deepcloud/salman/dataset/h5_data_all_chuncked \
    --save /mydata/deepcloud/yves/A_RadiativeFlux/results/rnn_1d_full_30percent_256 \
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
    --dropout 0.0 \
    --lstm-units 64 64 128 128 256 256 \
    --mlp-units 64 128 \
    --lstm-droprate 0.0 \
    --wandb-mode online

echo "RNN 1D Full Globe training completed!" 