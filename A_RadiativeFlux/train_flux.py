#!/usr/bin/env python3
"""
Unified training script for radiative flux prediction that supports:
- 1D (column-wise) and 3D (triangle-wise) modeling approaches
- Triangle and full dataset selection  
- Heating rate smoothness loss options
- All flux model types (ViT, AFNO, GNN, RNN, etc.)

Usage:
    # 1D models (column-wise)
    python unified_train_flux.py --model vit --mode 1d --dataset-type full
    python unified_train_flux.py --model gnn --mode 1d --dataset-type triangle --triangle-id 0
    
    # 3D models (triangle-wise)  
    python unified_train_flux.py --model gnn_3d --mode 3d --dataset-type triangle --grid-file-path /path/to/grid.nc
    
    # With heating rate smoothness loss
    python unified_train_flux.py --model vit --mode 1d --hr-smoothness-weight 0.1 --hr-smoothness-top-levels 10
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

from flux_data_loader import UnifiedFluxDataset
from utils.data_utils import DataNormalizer
from utils.flux_utils import calculate_heating_rates, HeatingRateSmoothnessLoss, SmoothHeatingRateSmoothnessLoss, EnergyConservationLossV2
from utils.evaluation_utils import process_timing_statistics, warm_up_model
#from flux_analysis.attention_analysis import run_attention_analysis_if_enabled

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

parser = argparse.ArgumentParser(description='Unified Flux Model Training')

# Core parameters
parser.add_argument('--dataset', type=str, required=True, help='Path to dataset')
parser.add_argument('--save', type=str, required=True, help='Path to save results')
parser.add_argument('--model', type=str, required=True, 
                    help='Model type: vit, afno, gnn, gnn_3d, rnn, unet, etc.')
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

# Training parameters
parser.add_argument('--train', action=argparse.BooleanOptionalAction, default=True, help='Enable training')
parser.add_argument('--test', action=argparse.BooleanOptionalAction, default=True, help='Enable testing')
parser.add_argument('--num-epoch', type=int, default=100, help='Number of training epochs')
parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'], help='Optimizer type')
parser.add_argument('--clip', type=float, default=1.0, help='Gradient clipping threshold')

# Model parameters
parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden/embedding dimension')
parser.add_argument('--layers', type=int, default=4, help='Number of model layers')
parser.add_argument('--heads', type=int, default=6, help='Number of attention heads (transformers)')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
parser.add_argument('--emb-dropout', type=float, default=0.0, help='Embedding dropout rate')

# Data-specific parameters
parser.add_argument('--num-cells', type=int, default=81920, help='Total number of ICON cells')
parser.add_argument('--channel-3d', type=int, default=6, help='Number of 3D input channels')
parser.add_argument('--channel-2d', type=int, default=6, help='Number of 2D input channels')
parser.add_argument('--channel-out', type=int, default=4, help='Number of output channels')
parser.add_argument('--height-in', type=int, default=70, help='Number of vertical levels')

# Triangle/region specific parameters
parser.add_argument('--triangle-id', type=int, default=0, help='Triangle ID for regional training')
parser.add_argument('--triangle-division-factor', type=int, default=1, help='Triangle division factor')

# 3D-specific parameters (for 3D models)
parser.add_argument('--grid-file-path', type=str, help='Path to ICON grid file (required for 3D models)')
parser.add_argument('--edge-channels-in', type=int, default=1, help='Number of edge feature channels (GNN)')
parser.add_argument('--fully-connected', action=argparse.BooleanOptionalAction, default=False, help='Use fully connected graph (GNN)')
parser.add_argument('--disable-horizontal', action=argparse.BooleanOptionalAction, default=False, help='Disable horizontal edges (GNN)')
parser.add_argument('--max-hops', type=int, default=1, help='Maximum hops for graph models')
parser.add_argument('--max-skip', type=int, default=3, help='Maximum skip distance (GNN)')

# Model-specific parameters
parser.add_argument('--patch-size', type=int, default=1, help='Patch size (ViT/AFNO)')
parser.add_argument('--dim-head', type=int, default=64, help='Dimension per attention head')
parser.add_argument('--mlp-ratio', type=float, default=4.0, help='MLP expansion ratio')

# AFNO specific
parser.add_argument('--afno-sparsity-threshold', type=float, default=0.01, help='AFNO sparsity threshold')
parser.add_argument('--hard-thresholding-fraction', type=float, default=1, help='AFNO hard thresholding fraction')
parser.add_argument('--fno-blocks', type=int, default=8, help='Number of FNO blocks')
parser.add_argument('--hidden-size-factor', type=int, default=1, help='AFNO hidden size factor')
parser.add_argument('--double-skip', action=argparse.BooleanOptionalAction, default=True, help='Use double skip connections')

# RNN specific
parser.add_argument('--lstm-units', nargs='+', type=int, default=[256, 512], help='LSTM units for RNN model')
parser.add_argument('--mlp-units', nargs='+', type=int, default=[256, 256], help='MLP units for RNN model')
parser.add_argument('--lstm-droprate', type=float, default=0.0, help='Dropout rate for LSTM layers')

# UNet specific
parser.add_argument('--cnn-units', nargs='+', type=int, default=[64, 128, 256, 512], help='CNN units for UNet')
parser.add_argument('--cnn-kernel-sizes', nargs='+', type=int, default=[2, 2, 2, 2], help='CNN kernel sizes for UNet')

# Heating Rate Loss parameters
parser.add_argument('--hr-smoothness-weight', type=float, default=0.0,
                   help='Weight for heating rate smoothness loss (0.0 to disable)')
parser.add_argument('--hr-smoothness-top-levels', type=int, default=None,
                   help='Apply heating rate smoothness only to top N levels')
parser.add_argument('--hr-smoothness-type', type=str, default='original', 
                   choices=['original', 'smooth'], 
                   help='Type of heating rate smoothness loss: original (hard cutoff) or smooth (gradual transition)')
parser.add_argument('--hr-smooth-transition', type=str, default='linear',
                   choices=['linear', 'sigmoid', 'quadratic'],
                   help='Transition type for smooth HR loss: linear, sigmoid, or quadratic')

# Energy Conservation Loss parameters  
parser.add_argument('--energy-conservation-weight', type=float, default=0.0,
                   help='Weight for energy conservation physics loss (0.0 to disable)')
parser.add_argument('--energy-conservation-alpha', type=float, default=0.5,
                   help='Balance between MSE (alpha) and energy conservation (1-alpha) in physics loss')

# Attention Analysis
parser.add_argument('--attention-analysis', action='store_true', help='Run attention analysis during testing (ViT models only)')
parser.add_argument('--attention-samples', type=int, default=500, help='Number of samples for attention analysis')
parser.add_argument('--attention-layers', nargs='+', type=int, default=None, help='Specific layers to analyze (e.g., 0 1 2 3)')

# Logging
parser.add_argument('--wandb-mode', type=str, default='disabled', choices=['online', 'offline', 'disabled'], help='W&B mode')

# Memory efficiency parameters
#parser.add_argument('--test-chunk-size', type=int, default=100, 
#                    help='Number of batches to process before saving during testing (for memory efficiency)')
#parser.add_argument('--memory-efficient-test', action=argparse.BooleanOptionalAction, default=False,
#                    help='Use memory-efficient testing that saves results in chunks')

args = parser.parse_args()

# Validate arguments
if args.mode == '3d' and args.grid_file_path is None:
    parser.error("--grid-file-path is required for 3D models")

if args.mode == '1d' and args.model == 'gnn_3d':
    parser.error(f"Model {args.model} is not compatible with 1D mode")

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
    
    # 1D Models
    if args.mode == '1d':
        if args.model == 'vit':
            from models_1d.vit import ViT
            model = ViT(
                patch_size=args.patch_size,
                dim=args.hidden_dim,
                mlp_dim=int(args.hidden_dim * args.mlp_ratio),
                depth=args.layers,
                heads=args.heads,
                channel_3d=args.channel_3d,
                channel_2d=args.channel_2d,
                channel_out=args.channel_out,
                height_in=args.height_in,
                dim_head=args.dim_head,
                dropout=args.dropout,
                emb_dropout=args.emb_dropout,
                device=device
            ).to(device)
            
        elif args.model == 'afno':
            from models_1d.afno import AFNONet
            model = AFNONet(
                patch_size=args.patch_size,
                embed_dim=args.hidden_dim,
                depth=args.layers,
                dropout=args.dropout,
                channel_3d=args.channel_3d,
                channel_2d=args.channel_2d,
                channel_out=args.channel_out,
                height_in=args.height_in,
                mlp_ratio=args.mlp_ratio,
                fno_blocks=args.fno_blocks,
                sparsity_threshold=args.afno_sparsity_threshold,
                hard_thresholding_fraction=args.hard_thresholding_fraction,
                hidden_size_factor=args.hidden_size_factor,
                double_skip=args.double_skip,
                device=device
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
                channel_2d=args.channel_2d,
                channels_out=args.channel_out,
                edge_channels_in=args.edge_channels_in,
                fully_connected=args.fully_connected,
                device=device
            ).to(device)
            
        elif args.model == 'rnn':
            from models_1d.rnn import FastRnnIg
            model = FastRnnIg(
                height_in=args.height_in,
                channel_out=args.channel_out,
                channel_3d=args.channel_3d,
                channel_2d=args.channel_2d,
                lstm_units=args.lstm_units,
                lstm_droprate=args.lstm_droprate,
                mlp_units=args.mlp_units,
                device=device
            ).to(device)
            
        elif args.model == 'unet':
            from models_1d.unet import UNet
            model = UNet(
                height_in=args.height_in,
                channel_3d=args.channel_3d,
                channel_2d=args.channel_2d,
                channel_out=args.channel_out,
                cnn_units=args.cnn_units,
                kernel_sizes=args.cnn_kernel_sizes,
                dropout=args.dropout,
                device=device
            ).to(device)
        else:
            raise ValueError(f"1D model {args.model} not supported")
    
    # 3D Models    
    elif args.mode == '3d':
        if args.model == 'gnn_3d':
            from models_3d.gnn_3d import GNN3d
            model = GNN3d(
                total_cols=args.num_cells,
                grid_file_path=args.grid_file_path,
                triangle_id=args.triangle_id,
                embed_dim=args.hidden_dim,
                depth=args.layers,
                dropout=args.dropout,
                channels_in_3d=args.channel_3d,
                channels_in_2d=args.channel_2d,
                channels_out=args.channel_out,
                edge_channels_in=args.edge_channels_in,
                num_height_levels=args.height_in,
                device=device,
                division_factor=args.triangle_division_factor,
                fully_connected=args.fully_connected,
                disable_horizontal=args.disable_horizontal
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


def create_data_loader(filenames, shuffle=False, num_workers=None):
    """Create unified data loader based on mode"""
    if num_workers is None:
        num_workers = args.num_workers
        
    dataset = UnifiedFluxDataset(
        filenames=filenames,
        mode=args.mode,
        dataset_type=args.dataset_type,
        triangle_id=args.triangle_id,
        division_factor=args.triangle_division_factor,
        shuffle=shuffle,
        subsample=args.subsample,
        cache_dir='/tmp',
        total_cols=args.num_cells,
        subsample_seed=seed,
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


def train_model(model, train_set, valid_set, normalizer):
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
        tags=['icon grid', 'flux', args.mode],
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
    
    # Initialize smoothness loss if enabled
    use_hr_smoothness = args.hr_smoothness_weight > 0
    if use_hr_smoothness:
        logger.info(f"Using heating rate smoothness loss with weight={args.hr_smoothness_weight}")
        
        if args.hr_smoothness_type == 'smooth':
            logger.info(f"Using smooth transition HR loss with {args.hr_smooth_transition} transition")
            hr_smoothness = SmoothHeatingRateSmoothnessLoss(
                weight=args.hr_smoothness_weight,
                top_levels=args.hr_smoothness_top_levels,
                mode=args.mode,
                transition_type=args.hr_smooth_transition
            ).to(device)
        else:
            logger.info("Using original HR loss with hard cutoff")
            hr_smoothness = HeatingRateSmoothnessLoss(
                weight=args.hr_smoothness_weight,
                top_levels=args.hr_smoothness_top_levels,
                mode=args.mode
            ).to(device)
            
    # Initialize energy conservation loss if enabled
    use_energy_conservation = args.energy_conservation_weight > 0
    if use_energy_conservation:
        logger.info(f"Using energy conservation physics loss with weight={args.energy_conservation_weight}, alpha={args.energy_conservation_alpha}")
        energy_conservation_loss = EnergyConservationLossV2(alpha=args.energy_conservation_alpha, mode=args.mode).to(device)

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
        
        smoothness_losses_train = []
        smoothness_losses_valid = []
        
        # Reset energy conservation loss if enabled
        if use_energy_conservation:
            energy_conservation_loss.reset()

        # Training step
        model.train(True)
        for i, data in enumerate(train_set):
            t1_1 = time.perf_counter()

            batch_x3, batch_x2, batch_y = data
            batch_x3, batch_x2, batch_y = (
                batch_x3.to(device), batch_x2.to(device), batch_y.to(device)
            )

            # Normalize inputs
            batch_x3_norm, batch_x2_norm, batch_x2_orig = normalizer.normalize(batch_x3, batch_x2)

            # Forward pass
            outputs = model(batch_x3_norm, batch_x2_norm, batch_x2_orig)
            
            # Calculate losses
            mse_loss = train_loss(outputs, batch_y)
            
            if use_hr_smoothness:
                smoothness_loss = hr_smoothness(outputs, batch_x3, batch_x2)
                loss = mse_loss + smoothness_loss
                smoothness_losses_train.append(smoothness_loss.item())
                
            # Add energy conservation loss if enabled
            if use_energy_conservation:
                energy_conservation_loss.update(batch_y, outputs, batch_x3, batch_x2)
                ec_components = energy_conservation_loss.get_components()
                ec_loss_value = energy_conservation_loss.compute()
                loss = mse_loss + args.energy_conservation_weight * ec_loss_value

            batch_mae = train_mae(outputs, batch_y)

            # Backward pass
            if i >= 0:
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
                vx3d, vx2d, v_labels = v_data
                vx3d, vx2d, v_labels = (
                    vx3d.to(device), vx2d.to(device), v_labels.to(device)
                )

                vx3d_norm, vx2d_norm, vx2d_orig = normalizer.normalize(vx3d, vx2d)
                v_outputs = model(vx3d_norm, vx2d_norm, vx2d_orig)
                
                if use_hr_smoothness:
                    v_smoothness_loss = hr_smoothness(v_outputs, vx3d, vx2d).item()
                    smoothness_losses_valid.append(v_smoothness_loss)
                    
                # Track energy conservation loss for validation if enabled
                if use_energy_conservation:
                    # Create a temporary loss instance for validation tracking
                    temp_ec_loss = EnergyConservationLossV2(alpha=args.energy_conservation_alpha, mode=args.mode).to(device)
                    temp_ec_loss.update(v_labels, v_outputs, vx3d, vx2d)

                valid_loss.update(v_outputs, v_labels)
                valid_mae.update(v_outputs, v_labels)

        # Compute metrics
        total_train_loss = train_loss.compute()
        total_valid_loss = valid_loss.compute()
        total_train_mae = train_mae.compute()
        total_valid_mae = valid_mae.compute()

        t2 = time.perf_counter()

        # Prepare log dictionary
        log_dict = {
            'epoch': epoch_number,
            'loss': total_train_loss,
            'val_loss': total_valid_loss,
            'mean_absolute_error': total_train_mae,
            'val_mean_absolute_error': total_valid_mae
        }
        
        # Add smoothness metrics if enabled
        if use_hr_smoothness and smoothness_losses_train:
            log_dict['hr_smoothness'] = sum(smoothness_losses_train) / len(smoothness_losses_train)
        if use_hr_smoothness and smoothness_losses_valid:
            log_dict['val_hr_smoothness'] = sum(smoothness_losses_valid) / len(smoothness_losses_valid)
            
        # Add energy conservation metrics if enabled
        if use_energy_conservation:
            ec_components = energy_conservation_loss.get_components()
            log_dict['energy_conservation_mse'] = ec_components['mse_component']
            log_dict['energy_conservation_physics'] = ec_components['ec_component']
            log_dict['energy_conservation_total'] = ec_components['total_loss']

        # Log to W&B
        wandb.log(log_dict)

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


def test_model(model, test_set, normalizer):
    logger.info('Testing started...')

    # Load best model
    best_chkpt = join(checkpoint_path, 'best_model.pth')
    assert isfile(best_chkpt), 'Best model checkpoint not found!'
    checkpoint = torch.load(best_chkpt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Run attention analysis if enabled
    #run_attention_analysis_if_enabled(
    #    model=model, 
    #    test_loader=test_set, 
    #    normalizer=normalizer, 
    #    save_dir=test_path, 
    #    num_samples=getattr(args, 'attention_samples', 500),
    #    enabled=getattr(args, 'attention_analysis', False),
    #    target_layers=getattr(args, 'attention_layers', None)
    #)

    # Initialize metrics
    test_loss = MeanSquaredError().to(device)
    test_mae = MeanAbsoluteError().to(device)
    
    use_hr_smoothness = args.hr_smoothness_weight > 0
    if use_hr_smoothness:
        if args.hr_smoothness_type == 'smooth':
            hr_smoothness = SmoothHeatingRateSmoothnessLoss(
                weight=1.0,
                top_levels=args.hr_smoothness_top_levels,
                mode=args.mode,
                transition_type=args.hr_smooth_transition
            ).to(device)
        else:
            hr_smoothness = HeatingRateSmoothnessLoss(
                weight=1.0,
                top_levels=args.hr_smoothness_top_levels,
                mode=args.mode
            ).to(device)
        smoothness_losses = []

   
    # Original approach - collect all results in memory
    y_true, y_pred = [], []
    h_true, h_pred = [], []
    
    # Timing configuration
    num_timing_batches = 50 if args.mode == '3d' else 50
    warmup_batches = 2 if args.mode == '3d' else 2
    timing_data = []
    
    # Perform warmup
    warm_up_model(model, test_set, normalizer, device, warmup_batches)

    t1 = time.perf_counter()
    model.eval()
    
    total_batches = 0
    
    with torch.no_grad():
        for i, data in enumerate(test_set):
            batch_x3, batch_x2, batch_y = data
            batch_x3, batch_x2, batch_y = (
                batch_x3.to(device), batch_x2.to(device), batch_y.to(device)
            )

            batch_x3_norm, batch_x2_norm, batch_x2_orig = normalizer.normalize(batch_x3, batch_x2)

            # Time inference for subset of batches
            if i < num_timing_batches:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                outputs = model(batch_x3_norm, batch_x2_norm, batch_x2_orig)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.perf_counter()
                timing_data.append(end - start)
            else:
                outputs = model(batch_x3_norm, batch_x2_norm, batch_x2_orig)
                
            # Track smoothness if enabled
            if use_hr_smoothness:
                smoothness_loss = hr_smoothness(outputs, batch_x3, batch_x2).item()
                smoothness_losses.append(smoothness_loss)

            # Calculate metrics
            loss = test_loss(outputs, batch_y)
            mae = test_mae(outputs, batch_y)
            
            # Collect results
            batch_y_cpu = batch_y.detach().cpu()
            outputs_cpu = outputs.detach().cpu()
            h_true_cpu = calculate_heating_rates(batch_y, batch_x3, batch_x2, mode=args.mode).detach().cpu()
            h_pred_cpu = calculate_heating_rates(outputs, batch_x3, batch_x2, mode=args.mode).detach().cpu()
            
            y_true.append(batch_y_cpu)
            y_pred.append(outputs_cpu)
            h_true.append(h_true_cpu)
            h_pred.append(h_pred_cpu)

            total_batches += 1
            
            if i % 100 == 99:
                print(f'batch {i+1}, loss: {loss:.4f}, mae: {mae:.4f}')

    t2 = time.perf_counter()


    # Original approach - save all at once
    y_true = torch.cat(y_true, 0)
    y_pred = torch.cat(y_pred, 0)
    h_true = torch.cat(h_true, 0)
    h_pred = torch.cat(h_pred, 0)

    logger.info(f'Saving results ({len(y_true)} samples)...')
    
    with open(join(test_path, 'y_true.pickle'), 'wb') as f:
        pickle.dump(y_true, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(join(test_path, 'y_pred.pickle'), 'wb') as f:
        pickle.dump(y_pred, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open(join(test_path, 'h_true.pickle'), 'wb') as f:
        pickle.dump(h_true, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(join(test_path, 'h_pred.pickle'), 'wb') as f:
        pickle.dump(h_pred, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f'Results saved to {test_path}')

    # Final metrics
    total_test_loss = test_loss.compute()
    total_test_mae = test_mae.compute()

    print(f'Test time: {t2-t1:.2f}s, loss: {total_test_loss:.4f}, mae: {total_test_mae:.4f}')
    
    if use_hr_smoothness and smoothness_losses:
        avg_smoothness = sum(smoothness_losses) / len(smoothness_losses)
        print(f'HR smoothness: {avg_smoothness:.4f}')

    # Process and save timing statistics
    if timing_data:
        timing_stats = process_timing_statistics(
            timing_data, model, test_path, args, count_parameters,
            test_loss=total_test_loss, test_mae=total_test_mae
        )


def main():
    logger.info('Unified Flux Training Started...')
    
    # Find and sort files
    FILENAMES = glob.glob(join(args.dataset, '*.h5'))
    time_indices = [float(re.search(r'_time_(.*?)\.h5', f).group(1)) for f in FILENAMES]
    sorted_files = [x for _, x in sorted(zip(time_indices, FILENAMES))]

    # Split data
    train_files = sorted_files[200:2000]
    val_files = sorted_files[:160] + sorted_files[2020:2180]
    test_files = sorted_files[2220:]

    # Subsample if requested
    if args.percent < 1:
        for file_list in [train_files, val_files, test_files]:
            indices = prng.choice(len(file_list), max(1, int(args.percent * len(file_list))), replace=False)
            file_list[:] = [file_list[i] for i in indices]

    # Load normalization parameters
    stats_file = join(args.dataset, 'normalizer_stats_per_feat.pickle')
    mean2d, var2d, mean3d, var3d = get_normalization_params(stats_file)
    normalizer = DataNormalizer(mean2d, var2d, mean3d, var3d, device=device)

    # Create model
    model = get_model()
    num_params = count_parameters(model)
    print(f"Model has {num_params:,} trainable parameters")

    # Training
    if args.train:
        train_loader = create_data_loader(train_files, shuffle=True)
        val_loader = create_data_loader(val_files, shuffle=False)
        train_model(model, train_loader, val_loader, normalizer)

    # Testing
    if args.test:
        test_loader = create_data_loader(test_files, shuffle=False)
        test_model(model, test_loader, normalizer)

    logger.info('Training completed!')


if __name__ == '__main__':
    main() 