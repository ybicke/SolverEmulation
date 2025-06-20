#!/usr/bin/env python3
"""
Unified training script for tendency prediction that supports both 1D and 3D modeling approaches.

Usage:
    # 1D models (column-wise)
    python unified_train_tendency.py --model vit_tendency --mode 1d --dataset-type triangle
    python unified_train_tendency.py --model gnn --mode 1d --dataset-type full
    
    # 3D models (triangle-wise)  
    python unified_train_tendency.py --model gnn_3d_tendency --mode 3d --dataset-type triangle
    python unified_train_tendency.py --model graph_transformer --mode 3d --dataset-type triangle
"""

import os
import re
import sys
import time
import glob
import yaml
import pickle
import logging
import random

import argparse
from os.path import join, dirname, basename, normpath, isfile

import wandb
import torch
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsoluteError, MeanSquaredError

from tendency_data_loader_simple import TendencyDataset
from utils.data_utils import (
    DataNormalizer, 
    interpolate_w_to_full_levels, 
    transform_targets, 
    inverse_transform_targets,
)
from utils.evaluation_utils import process_timing_statistics, warm_up_model, create_test_summary

sys.path.append(dirname(__file__))

# Set random seeds
seed = 42
wandb_config = {'seed': seed}
prng = np.random.RandomState(seed)

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='Unified Tendency Model Training')

# Core parameters
parser.add_argument('--dataset-input', type=str, required=True, help='Path to input dataset')
parser.add_argument('--dataset-output', type=str, required=True, help='Path to output dataset')
parser.add_argument('--save', type=str, required=True, help='Path to save results')
parser.add_argument('--model', type=str, required=True, 
                    help='Model type: vit_tendency, afno_tendency, gnn, gnn_3d, graph_transformer, etc.')
parser.add_argument('--mode', type=str, default='1d', choices=['1d', '3d'],
                    help='Training mode: 1d (column-wise) or 3d (triangle-wise)')
parser.add_argument('--dataset-type', type=str, default='triangle', choices=['triangle', 'full'],
                    help='Dataset selection: triangle (specific region) or full (all columns)')

# Data parameters
parser.add_argument('--percent', type=float, default=1.0, help='Percentage of data to use')
parser.add_argument('--subsample', type=float, default=None, help='Subsampling rate')
parser.add_argument('--num-workers', type=int, default=os.cpu_count(), help='Number of data loading workers')
parser.add_argument('--prefetch-factor', type=int, default=2, help='Prefetch factor for data loading')
parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
parser.add_argument('--shuffle', action=argparse.BooleanOptionalAction, default=True, help='Shuffle training data')
parser.add_argument('--shuffle-columns', action=argparse.BooleanOptionalAction, default=False, help='Shuffle column order within triangle (1D mode only)')

# Training parameters
parser.add_argument('--train', action=argparse.BooleanOptionalAction, default=True, help='Enable training')
parser.add_argument('--test', action=argparse.BooleanOptionalAction, default=True, help='Enable testing')
parser.add_argument('--num-epoch', type=int, default=100, help='Number of training epochs')
parser.add_argument('--learning-rate', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'], help='Optimizer type')
parser.add_argument('--clip', type=float, default=1.0, help='Gradient clipping threshold')

# Model parameters
parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden/embedding dimension')
parser.add_argument('--layers', type=int, default=4, help='Number of model layers')
parser.add_argument('--heads', type=int, default=6, help='Number of attention heads (for transformers)')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
parser.add_argument('--emb-dropout', type=float, default=0.0, help='Embedding dropout rate')

# Data-specific parameters
parser.add_argument('--num-cells', type=int, default=81920, help='Total number of ICON cells')
parser.add_argument('--channel-3d', type=int, default=10, help='Number of 3D input channels (including w)')
parser.add_argument('--channel-2d', type=int, default=3, help='Number of 2D input channels')
parser.add_argument('--channels-out', type=int, default=7, help='Number of output channels')
parser.add_argument('--height', type=int, default=70, help='Number of vertical levels')

# Triangle/region specific parameters
parser.add_argument('--triangle-id', type=int, default=39, help='Triangle ID for regional training')
parser.add_argument('--triangle-division-factor', type=int, default=4, help='Triangle division factor')

# 3D-specific parameters (for 3D models)
parser.add_argument('--grid-file-path', type=str, help='Path to ICON grid file (required for 3D models)')
parser.add_argument('--edge-channels-in', type=int, default=1, help='Number of edge feature channels (GNN)')
parser.add_argument('--fully-connected', action=argparse.BooleanOptionalAction, default=False, help='Use fully connected graph (GNN)')
parser.add_argument('--disable-horizontal', action=argparse.BooleanOptionalAction, default=False, help='Disable horizontal edges (GNN)')
parser.add_argument('--max-hops', type=int, default=1, help='Maximum hops for graph models')

# Model-specific parameters
parser.add_argument('--patch-size', type=int, default=1, help='Patch size (ViT/AFNO)')
parser.add_argument('--dim-head', type=int, default=32, help='Dimension per attention head')
parser.add_argument('--mlp-ratio', type=float, default=4.0, help='MLP expansion ratio')
parser.add_argument('--max-skip', type=int, default=3, help='Maximum skip distance (GNN)')

# Location features
parser.add_argument('--use-lonlat', action=argparse.BooleanOptionalAction, default=False, 
                    help='Include longitude and latitude coordinates as 2D features')

# AFNO specific
parser.add_argument('--afno-sparsity-threshold', type=float, default=0.01, help='AFNO sparsity threshold')
parser.add_argument('--hard-thresholding-fraction', type=float, default=1, help='AFNO hard thresholding fraction')
parser.add_argument('--fno-blocks', type=int, default=8, help='Number of FNO blocks')
parser.add_argument('--hidden-size-factor', type=int, default=1, help='AFNO hidden size factor')

# Logging
parser.add_argument('--wandb-mode', type=str, default='disabled', choices=['online', 'offline', 'disabled'], help='W&B mode')

# Normalization files
parser.add_argument('--input-stats-file', type=str, 
                    default='/mydata/deepcloud/shared/h5_tendency_all/normalizer_stats_per_feat_updated.pickle',
                    help='Path to input normalization statistics file')
parser.add_argument('--target-stats-file', type=str,
                    default='/mydata/deepcloud/shared/h5_tendency_all/normalizer_stats_per_feat_y2_no_temp.pickle',
                    help='Path to target normalization statistics file')

args = parser.parse_args()

# Validate arguments
if args.mode == '3d' and args.grid_file_path is None:
    parser.error("--grid-file-path is required for 3D models")

if args.use_lonlat and args.grid_file_path is None:
    parser.error("--grid-file-path is required when --use-lonlat is enabled")

if args.mode == '1d' and args.model.endswith('_3d') or args.model.startswith('graph_'):
    parser.error(f"Model {args.model} is not compatible with 1D mode")

# Calculate effective 2D channel count (original + lon/lat if enabled)
effective_channel_2d = args.channel_2d + (2 if args.use_lonlat else 0)

# Setup paths and logging
save_id = f'{basename(normpath(args.save))}'
checkpoint_path = join(args.save)
os.makedirs(checkpoint_path, exist_ok=True)
test_path = join(args.save, 'test/')
os.makedirs(test_path, exist_ok=True)

logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f'Using device: {device}, CPUs: {os.cpu_count()}')

if args.use_lonlat:
    logger.info(f"Using lon/lat features: 2D channels increased from {args.channel_2d} to {effective_channel_2d}")

# Save configuration
with open(join(args.save, 'config.yml'), 'w') as f:
    yaml.dump(args.__dict__, f, default_flow_style=False)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_normalization_params(stats_file):
    with open(stats_file, 'rb') as f:
        stats = pickle.load(f)
        return (
            torch.tensor(stats['mean2d'], dtype=torch.float32).to(device),
            torch.tensor(stats['var2d'], dtype=torch.float32).to(device),
            torch.tensor(stats['mean3d'], dtype=torch.float32).to(device),
            torch.tensor(stats['var3d'], dtype=torch.float32).to(device)
        )


def get_model():
    """Create model based on args.model and args.mode"""
    logger.info(f'Creating {args.model} model for {args.mode} mode...')
    
    # Calculate effective 2D channel count (original + lon/lat if enabled)
    effective_channel_2d = args.channel_2d + (2 if args.use_lonlat else 0)
    
    # 1D Models
    if args.mode == '1d':
        if args.model == 'vit':
            from models_1d.vit import ViT
            model = ViT(
                patch_size=args.patch_size,
                embed_dim=args.hidden_dim,
                depth=args.layers,
                heads=args.heads,
                dropout=args.dropout,
                emb_dropout=args.emb_dropout,
                channels_in_3D=args.channel_3d,
                channels_in_2D=effective_channel_2d,
                channels_out=args.channels_out,
                height=args.height,
                mlp_ratio=args.mlp_ratio,
                dim_head=args.dim_head,
            ).to(device)
            
        elif args.model == 'afno':
            from models_1d.afno import AFNONet
            model = AFNONet(
                patch_size=args.patch_size,
                num_cells=args.num_cells,
                embed_dim=args.hidden_dim,
                depth=args.layers,
                dropout=args.dropout,
                emb_dropout=args.emb_dropout,
                channels_in_3D=args.channel_3d,
                channels_in_2D=effective_channel_2d,
                channels_out=args.channels_out,
                height=args.height,
                mlp_ratio=args.mlp_ratio,
                hard_thresholding_fraction=args.hard_thresholding_fraction,
                sparsity_threshold=args.afno_sparsity_threshold,
                fno_blocks=args.fno_blocks,
                hidden_size_factor=args.hidden_size_factor,
            ).to(device)
            
        elif args.model == 'gnn':
            from models_1d.gnn import AtmosphericColumnGNN
            model = AtmosphericColumnGNN(
                embed_dim=args.hidden_dim,
                depth=args.layers,
                dropout=args.dropout,
                max_skip=args.max_skip,
                emb_dropout=args.emb_dropout,
                channel_3d=args.channel_3d,
                channel_2d=effective_channel_2d,
                channels_out=args.channels_out,
                edge_channels_in=args.edge_channels_in,
                fully_connected=args.fully_connected,
                device=device
            ).to(device)
        else:
            raise ValueError(f"1D model {args.model} not supported")
    
    # 3D Models    
    elif args.mode == '3d':
        if args.model == 'gnn_3d':
            from models_3d.gnn_3d import GNN3D
            model = GNN3D(
                total_cols=args.num_cells,
                grid_file_path=args.grid_file_path,
                triangle_id=args.triangle_id,
                embed_dim=args.hidden_dim,
                depth=args.layers,
                dropout=args.dropout,
                channels_in_3d=args.channel_3d,
                channels_in_2d=effective_channel_2d,
                channels_out=args.channels_out,
                edge_channels_in=args.edge_channels_in,
                num_height_levels=args.height,
                device=device,
                division_factor=args.triangle_division_factor,
                fully_connected=args.fully_connected,
                disable_horizontal=args.disable_horizontal
            ).to(device)
            
        elif args.model == 'gt':
            from models_3d.gt_3d import GraphTransformer3D
            model = GraphTransformer3D(
                total_cols=args.num_cells,
                grid_file_path=args.grid_file_path,
                triangle_id=args.triangle_id,
                embed_dim=args.hidden_dim,
                depth=args.layers,
                dropout=args.dropout,
                channels_in_3d=args.channel_3d,
                channels_in_2d=effective_channel_2d,
                channels_out=args.channels_out,
                num_height_levels=args.height,
                device=device,
                division_factor=args.triangle_division_factor,
                heads=args.heads,
                dim_head=args.dim_head,
                mlp_ratio=args.mlp_ratio,
                fully_connected=args.fully_connected,
                disable_horizontal=args.disable_horizontal,
                max_hops=args.max_hops
            ).to(device)
            
        elif args.model == 'gt_simplified':
            from models_3d.gt_3d_simplified import SimplifiedGraphTransformer3D
            model = SimplifiedGraphTransformer3D(
                total_cols=args.num_cells,
                grid_file_path=args.grid_file_path,
                triangle_id=args.triangle_id,
                embed_dim=args.hidden_dim,
                depth=args.layers,
                dropout=args.dropout,
                channels_in_3d=args.channel_3d,
                channels_in_2d=effective_channel_2d,
                channels_out=args.channels_out,
                num_height_levels=args.height,
                device=device,
                division_factor=args.triangle_division_factor,
                heads=args.heads,
                dim_head=args.dim_head,
                mlp_ratio=args.mlp_ratio,
                fully_connected=args.fully_connected,
                disable_horizontal=args.disable_horizontal,
                max_hops=args.max_hops
            ).to(device)
            
        elif args.model == 'gt_enhanced':
            from models_3d.gt_3d_enhanced import EnhancedGraphTransformer3D
            model = EnhancedGraphTransformer3D(
                total_cols=args.num_cells,
                grid_file_path=args.grid_file_path,
                triangle_id=args.triangle_id,
                embed_dim=args.hidden_dim,
                depth=args.layers,
                dropout=args.dropout,
                channels_in_3d=args.channel_3d,
                channels_in_2d=effective_channel_2d,
                channels_out=args.channels_out,
                num_height_levels=args.height,
                device=device,
                division_factor=args.triangle_division_factor,
                heads=args.heads,
                dim_head=args.dim_head,
                mlp_ratio=args.mlp_ratio,
                fully_connected=args.fully_connected,
                disable_horizontal=args.disable_horizontal,
                max_hops=args.max_hops
            ).to(device)

            
        elif args.model == 'gt_gencast_style':
            from models_3d.gt_3d_gencast_style import GenCastStyleGraphTransformer3D
            model = GenCastStyleGraphTransformer3D(
                total_cols=args.num_cells,
                grid_file_path=args.grid_file_path,
                triangle_id=args.triangle_id,
                embed_dim=args.hidden_dim,
                depth=args.layers,
                dropout=args.dropout,
                channels_in_3d=args.channel_3d,
                channels_in_2d=effective_channel_2d,
                channels_out=args.channels_out,
                num_height_levels=args.height,
                device=device,
                division_factor=args.triangle_division_factor,
                heads=args.heads,
                dim_head=args.dim_head,
                mlp_ratio=args.mlp_ratio,
                fully_connected=args.fully_connected,
                disable_horizontal=args.disable_horizontal,
                max_hops=args.max_hops
            ).to(device)
            
        else:
            raise ValueError(f"3D model {args.model} not supported")
    
    return model


def find_latest_checkpoint(directory):
    ckpts = glob.glob(join(directory, f'checkpoint_epoch_*.pth'))
    if len(ckpts) == 0:
        return None, None
    idx = [int(re.search('checkpoint_epoch_(.*?).pth', cp).group(1)) for cp in ckpts]
    return join(directory, f'checkpoint_epoch_{max(idx)}.pth'), max(idx)


def create_data_loader(input_files, output_files, shuffle=False, num_workers=None):
    """Create unified data loader based on mode"""
    if num_workers is None:
        num_workers = args.num_workers
        
    dataset = TendencyDataset(
        input_filenames=input_files,
        output_filenames=output_files,
        mode=args.mode,
        dataset_type=args.dataset_type,
        triangle_id=args.triangle_id,
        division_factor=args.triangle_division_factor,
        shuffle=shuffle,
        shuffle_columns=args.shuffle_columns,
        subsample=args.subsample,
        total_cols=args.num_cells,
        use_lonlat=args.use_lonlat,
        grid_file_path=args.grid_file_path if args.use_lonlat else None,
    )

    dataloader_args = {
        'dataset': dataset,
        'batch_size': args.batch_size,
        'pin_memory': True,
        'num_workers': num_workers,
    }

    if num_workers > 0:
        dataloader_args['prefetch_factor'] = args.prefetch_factor

    return DataLoader(**dataloader_args)


def train_model(model, train_set, valid_set, normalizer, target_means, target_vars):
    logger.info('Training started...')
    
    # Initialize W&B
    wandb.init(
        project='deepcloud-yves',
        name=save_id,
        id=save_id,
        config={**wandb_config, **args.__dict__},
        sync_tensorboard=True,
        save_code=True,
        resume='allow',
        tags=['icon grid', 'tendency', args.mode],
        mode=args.wandb_mode
    )
    wandb.watch(model, log_freq=100)

    # Setup optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            eps=1e-8,
            weight_decay=0.01
        )

    # Initialize metrics
    train_loss = MeanSquaredError().to(device)
    valid_loss = MeanSquaredError().to(device)
    train_mae = MeanAbsoluteError().to(device)
    valid_mae = MeanAbsoluteError().to(device)

    # Load checkpoint if available
    p_path, cp_id = find_latest_checkpoint(checkpoint_path)
    if p_path is not None:
        logger.info(f'Loading checkpoint: {cp_id}, {p_path}')
        checkpoint = torch.load(p_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        init_epoch = cp_id
    else:
        init_epoch = 0

    best_loss = 1e9999999

    # Training loop
    for epoch in range(init_epoch, args.num_epoch):
        t1 = time.perf_counter()
        epoch_number = epoch + 1

        # Reset metrics
        train_mae.reset()
        valid_mae.reset()
        train_loss.reset()
        valid_loss.reset()

        # Training step
        model.train(True)
        for i, data in enumerate(train_set):
            t1_1 = time.perf_counter()

            batch_x3, batch_x2, batch_y, batch_w = data
            batch_x3, batch_x2, batch_y, batch_w = (
                batch_x3.to(device), batch_x2.to(device), 
                batch_y.to(device), batch_w.to(device)
            )

            # Interpolate w and concatenate with 3D data
            w_full = interpolate_w_to_full_levels(batch_w, mode=args.mode)
            batch_x3_with_w = torch.cat([batch_x3, w_full], dim=-1)

            # Normalize inputs
            batch_x3_norm, batch_x2_norm, _ = normalizer.normalize(batch_x3_with_w, batch_x2)

            # Transform targets
            batch_y_transformed = transform_targets(batch_y, target_means, target_vars, mode=args.mode)

            # Forward pass
            outputs = model(batch_x3_norm, batch_x2_norm)
            loss = train_loss(outputs, batch_y_transformed)
            batch_mae = train_mae(outputs, batch_y_transformed)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            t2_1 = time.perf_counter()
            if i % 100 == 99:
                print(f'batch {i+1}, time:{t2_1-t1_1:.3f}, '
                      f'loss: {loss:.4f}, mae: {batch_mae:.4f}')

        # Validation step
        model.eval()
        with torch.no_grad():
            for i, v_data in enumerate(valid_set):
                vx3d, vx2d, v_labels, v_w = v_data
                vx3d, vx2d, v_labels, v_w = (
                    vx3d.to(device), vx2d.to(device), 
                    v_labels.to(device), v_w.to(device)
                )

                # Process validation data same as training
                w_full = interpolate_w_to_full_levels(v_w, mode=args.mode)
                vx3d_with_w = torch.cat([vx3d, w_full], dim=-1)
                vx3d_norm, vx2d_norm, _ = normalizer.normalize(vx3d_with_w, vx2d)
                v_labels_transformed = transform_targets(v_labels, target_means, target_vars, mode=args.mode)

                v_outputs = model(vx3d_norm, vx2d_norm)
                valid_loss.update(v_outputs, v_labels_transformed)
                valid_mae.update(v_outputs, v_labels_transformed)

        # Compute metrics
        total_train_loss = train_loss.compute()
        total_valid_loss = valid_loss.compute()
        total_train_mae = train_mae.compute()
        total_valid_mae = valid_mae.compute()

        t2 = time.perf_counter()

        # Log to W&B
        wandb.log({
            'epoch': epoch_number,
            'loss': total_train_loss,
            'val_loss': total_valid_loss,
            'mean_absolute_error': total_train_mae,
            'val_mean_absolute_error': total_valid_mae
        })

        # Print progress
        print(f'{epoch_number:03}/{args.num_epoch}: '
              f'time: {t2-t1:.3f}, '
              f'loss: {total_train_loss:.4f}, '
              f'mae: {total_train_mae:.4f}, '
              f'val_loss: {total_valid_loss:.4f}, '
              f'val_mae: {total_valid_mae:.4f}')

        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_valid_loss,
        }
        torch.save(checkpoint, join(checkpoint_path, f'checkpoint_epoch_{epoch_number}.pth'))
        
        if total_valid_loss < best_loss:
            torch.save(checkpoint, join(checkpoint_path, 'best_model.pth'))
            best_loss = total_valid_loss

    return model


def test_model(model, test_set, normalizer, target_means, target_vars):
    logger.info('Testing started...')

    # Load best model
    best_chkpt = join(checkpoint_path, 'best_model.pth')
    assert isfile(best_chkpt), 'Best model checkpoint not found!'
    checkpoint = torch.load(best_chkpt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Initialize metrics
    test_loss = MeanSquaredError().to(device)
    test_mae = MeanAbsoluteError().to(device)

    y_true, y_pred = [], []
    
    # Timing configuration
    num_timing_batches = 10 if args.mode == '3d' else 10
    warmup_batches = 2 if args.mode == '3d' else 2
    timing_data = []
    
    # Perform warmup
    actual_warmup_batches = warm_up_model(model, test_set, normalizer, target_means, target_vars, device, args.mode, warmup_batches)
    
    t1 = time.perf_counter()
    total_batches = 0

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_set):
            batch_x3, batch_x2, batch_y, batch_w = data
            batch_x3, batch_x2, batch_y, batch_w = (
                batch_x3.to(device), batch_x2.to(device), 
                batch_y.to(device), batch_w.to(device)
            )

            # Process test data
            w_full = interpolate_w_to_full_levels(batch_w, mode=args.mode)
            batch_x3_with_w = torch.cat([batch_x3, w_full], dim=-1)
            batch_x3_norm, batch_x2_norm, _ = normalizer.normalize(batch_x3_with_w, batch_x2)
            batch_y_transformed = transform_targets(batch_y, target_means, target_vars, mode=args.mode)

            # Time inference for subset of batches
            if i < num_timing_batches:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                outputs = model(batch_x3_norm, batch_x2_norm)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.perf_counter()
                timing_data.append(end - start)
            else:
                outputs = model(batch_x3_norm, batch_x2_norm)

            # Calculate metrics
            loss = test_loss(outputs, batch_y_transformed)
            mae = test_mae(outputs, batch_y_transformed)

            # Inverse transform for saving
            outputs_original = inverse_transform_targets(outputs, target_means, target_vars, mode=args.mode)
            
            # Collect results
            y_true.append(batch_y.detach().cpu())
            y_pred.append(outputs_original.detach().cpu())
            
            total_batches += 1

            if i % 100 == 99:
                print(f'batch {i+1}, loss: {loss:.4f}, mae: {mae:.4f}')

    t2 = time.perf_counter()

    # Final metrics
    total_test_loss = test_loss.compute()
    total_test_mae = test_mae.compute()

    print(f'Test time: {t2-t1:.2f}s, loss: {total_test_loss:.4f}, mae: {total_test_mae:.4f}')

    # Save results
    y_true = torch.cat(y_true, 0)
    y_pred = torch.cat(y_pred, 0)
    total_samples = len(y_true)

    with open(join(test_path, 'y_true.pickle'), 'wb') as f:
        pickle.dump(y_true, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(join(test_path, 'y_pred.pickle'), 'wb') as f:
        pickle.dump(y_pred, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(join(test_path, 'target_stats.pickle'), 'wb') as f:
        pickle.dump({'means': target_means.cpu(), 'vars': target_vars.cpu()}, 
                   f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f'Results saved to {test_path}')
    
    # Process and save timing statistics
    if timing_data:
        timing_stats = process_timing_statistics(
            timing_data, model, test_path, args, count_parameters,
            test_loss=total_test_loss, test_mae=total_test_mae,
            num_warmup_batches=actual_warmup_batches, total_test_batches=total_batches
        )
    
    # Create comprehensive test summary
    create_test_summary(
        test_path, args, model, count_parameters, 
        total_test_loss, total_test_mae, t2-t1, total_samples
    )


def main():
    logger.info('Unified Tendency Training Started...')
    
    # Find and sort files
    INPUT_FILENAMES = glob.glob(join(args.dataset_input, '*_inputs_*.h5'))
    OUTPUT_FILENAMES = glob.glob(join(args.dataset_output, '*_tendencies_*.h5'))

    # Sort files by time
    input_time_indices = [float(re.search(r'_time_(\d+\.\d+)\.h5', f).group(1)) for f in INPUT_FILENAMES]
    output_time_indices = [float(re.search(r'_time_(\d+\.\d+)\.h5', f).group(1)) for f in OUTPUT_FILENAMES]

    sorted_input_time_indices = sorted(input_time_indices)
    sorted_output_time_indices = sorted(output_time_indices)

    assert sorted_input_time_indices == sorted_output_time_indices, "Time indices mismatch"

    sorted_input_files = [INPUT_FILENAMES[input_time_indices.index(t)] for t in sorted_input_time_indices]
    sorted_output_files = [OUTPUT_FILENAMES[output_time_indices.index(t)] for t in sorted_output_time_indices]

    # Split data
    train_input_files = sorted_input_files[200:2000]
    train_output_files = sorted_output_files[200:2000]
    val_input_files = sorted_input_files[:160] + sorted_input_files[2020:2180]
    val_output_files = sorted_output_files[:160] + sorted_output_files[2020:2180]
    test_input_files = sorted_input_files[2220:]
    test_output_files = sorted_output_files[2220:]

    # Subsample if requested
    if args.percent < 1:
        for file_list_pair in [(train_input_files, train_output_files), 
                              (val_input_files, val_output_files), 
                              (test_input_files, test_output_files)]:
            indices = prng.choice(len(file_list_pair[0]), 
                                max(1, int(args.percent * len(file_list_pair[0]))), 
                                replace=False)
            file_list_pair[0][:] = [file_list_pair[0][i] for i in indices]
            file_list_pair[1][:] = [file_list_pair[1][i] for i in indices]

    # Load normalization parameters
    mean2d, var2d, mean3d, var3d = get_normalization_params(args.input_stats_file)
    normalizer = DataNormalizer(mean2d, var2d, mean3d, var3d, device=device)

    # Load target statistics
    with open(args.target_stats_file, 'rb') as f:
        target_stats = pickle.load(f)

    target_means = torch.tensor(target_stats['mean'], dtype=torch.float32).to(device)
    target_vars = torch.tensor(target_stats['var'], dtype=torch.float32).to(device)

    # Create model
    model = get_model()
    num_params = count_parameters(model)
    print(f"Model has {num_params:,} trainable parameters")

    # Training
    if args.train:
        train_loader = create_data_loader(train_input_files, train_output_files, shuffle=True)
        val_loader = create_data_loader(val_input_files, val_output_files, shuffle=False)
        train_model(model, train_loader, val_loader, normalizer, target_means, target_vars)

    # Testing
    if args.test:
        test_loader = create_data_loader(test_input_files, test_output_files, shuffle=False)
        test_model(model, test_loader, normalizer, target_means, target_vars)

    logger.info('Training completed!')


if __name__ == '__main__':
    main() 