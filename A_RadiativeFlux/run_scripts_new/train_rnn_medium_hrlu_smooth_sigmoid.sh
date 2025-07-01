#!/bin/bash

# Set to fail on error
set -e

# Source setup script
source /mydata/deepcloud/yves/SolverEmulation/A_RadiativeFlux/run_scripts_new/setup.sh

myproject="deepcloud"
myusername="yves"
myfolder=/mydata/deepcloud/yves/SolverEmulation/A_RadiativeFlux/

cd $myfolder
echo "RNN training started (WITH smooth sigmoid transition HR loss)!"

# Run the training script with smooth sigmoid transition HR loss
python train_flux.py \
    --model rnn \
    --mode 1d \
    --dataset-type full \
    --dataset /mydata/deepcloud/salman/dataset/h5_data_all_chunck_time_triangle_id_20/ \
    --save /mydata/deepcloud/yves/A_RadiativeFlux/results/rnn_medium_hrlu_smooth_sigmoid \
    --batch-size 32 \
    --num-epoch 100 \
    --learning-rate 1e-3 \
    --optimizer adamw \
    --height-in 71 \
    --channel-3d 6 \
    --channel-2d 6 \
    --channel-out 4 \
    --lstm-units 256 512 \
    --mlp-units 256 256 \
    --lstm-droprate 0.0 \
    --hr-smoothness-weight 0.005 \
    --hr-smoothness-top-levels 30 \
    --hr-smoothness-type smooth \
    --hr-smooth-transition sigmoid \
    --wandb-mode disabled 