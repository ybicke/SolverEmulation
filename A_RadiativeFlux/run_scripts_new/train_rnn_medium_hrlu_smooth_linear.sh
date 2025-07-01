#!/bin/bash

# Set to fail on error
set -e

# Source setup script
source /mydata/deepcloud/yves/SolverEmulation/A_RadiativeFlux/run_scripts_new/setup.sh

myproject="deepcloud"
myusername="yves"
myfolder=/mydata/deepcloud/yves/SolverEmulation/A_RadiativeFlux/

cd $myfolder
echo "RNN training started (WITH smooth linear transition HR loss)!"

# Run the training script with smooth transition HR loss
python train_flux.py \
    --model rnn \
    --mode 1d \
    --dataset-type full \
    --dataset /mydata/deepcloud/salman/dataset/h5_data_all_chunck_time_triangle_id_20/ \
    --save /mydata/deepcloud/yves/A_RadiativeFlux/results/rnn_medium_hrlu_smooth_linear \
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
    --hr-smoothness-type smooth \
    --hr-smooth-transition linear \
    --lstm-droprate 0.0 \
    --wandb-mode online
 

