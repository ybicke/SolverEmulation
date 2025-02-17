#!/bin/bash

set -e
source /mydata/deepcloud/yves/SolverEmulation/bash_scripts/setup.sh

myfolder=/mydata/deepcloud/yves/SolverEmulation
cd $myfolder

echo "Training GNN GAT with 4 layers and 4 heads started!"

python train_column.py \
    --model gnn_gat_column \
    --dataset /mydata/deepcloud/salman/dataset/h5_data_all_chuncked \
    --save /mydata/deepcloud/yves/results_git/gnn_gat_Emb128_L4_H4 \
    --percent 0.1 \
    --subsample 0.1 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --num-cells 81920 \
    --train \
    --test \
    --shuffle \
    --batch-size 2048 \
    --vbatch 1 \
    --optimizer adamw \
    --clip 1.0 \
    --num-epoch 50 \
    --learning-rate 0.0005 \
    --patch-size 1 \
    --hidden-dim 128 \
    --layers 4 \
    --heads 4 \
    --vit-dropout 0.0 \
    --wandb-mode online 