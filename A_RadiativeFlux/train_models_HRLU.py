import os
import re
import sys
import time
import glob
import yaml
import pickle
import logging
import random
import math
import warnings


import argparse
from os.path import join, dirname, basename, normpath, isfile, exists

import wandb
import torch
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsoluteError, MeanSquaredError



from data_loader import IconColumnIterableDataset
from data_utils import DataNormalizer



sys.path.append(dirname(__file__))

PROFILE_NAME = 'default'
ENDPOING_URL = 'https://os.zhdk.cloud.switch.ch'  # SWITCH S3 server
BUCKET = 'deepcloud'
CACHE_DIR = '/tmp' # tempfile.TemporaryDirectory(dir='/tmp').name


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

import argparse
import os

parser = argparse.ArgumentParser(description='Train Transformer models.')

# General parameters
parser.add_argument('--model', type=str, default='vit', help='Name of the model to be trained')
parser.add_argument('--dataset', type=str, required=True, help='Path to dataset')
parser.add_argument('--save', type=str, required=True, help='Path to save the result')
parser.add_argument('--percent', type=float, default=1.0, help='Percent of data to train on')
parser.add_argument('--subsample', type=float, default=None, help='Subsampling rate')
parser.add_argument('--num-workers', type=int, default=os.cpu_count(), help='Number of workers to load data')
parser.add_argument('--prefetch-factor', type=int, default=2, help='Prefetch factor')
parser.add_argument('--wandb-mode', type=str, default='disabled', choices={'online', 'offline', 'disabled'}, help='Operating mode for W&B')
parser.add_argument('--num-cells', type=int, default=81921, help='Number of ICON cells in a complete Earth model')
parser.add_argument('--train', action=argparse.BooleanOptionalAction, default=True, help='Specify if training takes place')
parser.add_argument('--test', action=argparse.BooleanOptionalAction, default=True, help='Specify if test takes place')
parser.add_argument('--shuffle', action=argparse.BooleanOptionalAction, default=True, help='Shuffling the train dataset')
parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer')
parser.add_argument('--clip', type=float, default=1.0, help='Gradient clipping')
parser.add_argument('--num-epoch', type=int, default=100, help='Number of epochs')
parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')

# Common model parameters used by most models
parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension for all models')
parser.add_argument('--layers', type=int, default=4, help='Number of layers for all models')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate for all models')
parser.add_argument('--channel-out', type=int, default=4, help='Output channels for all models')
parser.add_argument('--channel-3d', type=int, default=6, help='3D input channels')
parser.add_argument('--channel-2d', type=int, default=6, help='2D input channels')
parser.add_argument('--patch-size', type=int, default=2, help='Patch size for transformer models')
parser.add_argument('--scale-output', action=argparse.BooleanOptionalAction, default=True, help='Whether to scale output')
parser.add_argument('--height-in', type=int, default=70, help='Number of height levels')

# Add this argument to your parser arguments section
parser.add_argument('--hr-smoothness-weight', type=float, default=0.0, 
                   help='Weight for heating rate smoothness loss (0.0 to disable, higher values create smoother profiles)')
parser.add_argument('--hr-smoothness-top-levels', type=int, default=None,
                   help='If set, apply heating rate smoothness only to the N uppermost vertical levels')

# To create argument groups, just call the method on the parser
# These groups are for better help text organization, but all arguments are still part of the main namespace

# ViT specific parameters
vit_group = parser.add_argument_group('ViT model arguments')
vit_group.add_argument('--heads', type=int, default=6, help='Number of attention heads')
vit_group.add_argument('--dim-head', type=int, default=64, help='Dimension of each attention head')
vit_group.add_argument('--emb-dropout', type=float, default=0.0, help='Dropout rate for embeddings')

# AFNO specific parameters
afno_group = parser.add_argument_group('AFNO model arguments')
afno_group.add_argument('--afno-sparsity-threshold', type=float, default=0.01, help='Sparsity threshold for AFNO')
afno_group.add_argument('--hard-thresholding-fraction', type=float, default=1, help='Hard thresholding fraction AFNO')
afno_group.add_argument('--cutoff-frequency', type=float, default=0.1, help='Cutoff frequency low pass filtering in AFNO')
afno_group.add_argument('--zero-freq-indices', nargs='+', type=int, default=None, help='Zero frequency indices to zero out')
afno_group.add_argument('--fno-blocks', type=int, default=8, help='Number of blocks in AFNO1D filter')
afno_group.add_argument('--hidden-size-factor', type=int, default=1, help='Expansion factor for FFT dimension')
afno_group.add_argument('--double-skip', action=argparse.BooleanOptionalAction, default=True, help='Use double skip connections')
afno_group.add_argument('--mlp-ratio', type=float, default=4.0, help='Ratio of MLP hidden dim to embedding dim')

# GNN specific parameters
gnn_group = parser.add_argument_group('GNN model arguments')
gnn_group.add_argument('--max-skip', type=int, default=3, help='Maximum skip distance for hierarchical edges')
gnn_group.add_argument('--fully-connected', action=argparse.BooleanOptionalAction, default=False, help='Use fully connected graph')
gnn_group.add_argument('--edge-channels-in', type=int, default=1, help='Number of edge feature channels')
gnn_group.add_argument('--heights-file', type=str, default=None, help='Path to file containing height data')

# RNN specific parameters
rnn_group = parser.add_argument_group('RNN model arguments')
rnn_group.add_argument('--lstm-units', nargs='+', type=int, default=[256, 512], help='LSTM units for RNN model')
rnn_group.add_argument('--mlp-units', nargs='+', type=int, default=[256, 256], help='MLP units for RNN model')
rnn_group.add_argument('--lstm-droprate', type=float, default=0.0, help='Dropout rate for LSTM layers')


args = parser.parse_args()


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
logger.info(f'Torch is using: {device}, number of cpus: {os.cpu_count()}')

with open(join(args.save, 'config.yml'), 'w') as f:
    yaml.dump(args.__dict__, f, default_flow_style=False)



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_normalization_params(stats_file):
    with open(stats_file, 'rb') as f:
        stats = pickle.load(f)
        return torch.tensor(stats['mean2d']).to(device), \
            torch.tensor(stats['var2d']).to(device), \
                torch.tensor(stats['mean3d']).to(device), \
                    torch.tensor(stats['var3d']).to(device)
    
def get_model(model_name):
    logger.info('Preparing the model...')
    
    if model_name == 'vit':
        from models.vit import ViT
        model = ViT(
            patch_size=args.patch_size,
            dim=args.hidden_dim,
            mlp_dim=args.hidden_dim,
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
        
        
    elif model_name == 'afno':
        from models.afno import AFNONet
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
        
    
    elif model_name == 'gnn':
        from models.gnn import AtmosphericColumnGNN
        
        model = AtmosphericColumnGNN(
            embed_dim=args.hidden_dim,
            depth=args.layers, 
            dropout=args.dropout,
            max_skip=args.max_skip,
            emb_dropout=args.dropout,  # Using main dropout for embedding
            channel_3d=args.channel_3d,
            channel_2d=args.channel_2d,
            channels_out=args.channel_out,
            edge_channels_in=args.edge_channels_in,
            fully_connected=args.fully_connected,
            device=device
        ).to(device)    
        
        
    elif model_name == 'gnn_optimized1':
        from models.gnn_optimized1 import AtmosphericColumnGNN
        
        model = AtmosphericColumnGNN(
            embed_dim=args.hidden_dim,
            depth=args.layers, 
            dropout=args.dropout,
            max_skip=args.max_skip,
            emb_dropout=args.dropout,  # Using main dropout for embedding
            channel_3d=args.channel_3d,
            channel_2d=args.channel_2d,
            channels_out=args.channel_out,
            edge_channels_in=args.edge_channels_in,
            fully_connected=args.fully_connected,
            device=device
        ).to(device)    
        
        
    elif model_name == 'gnn_broadcast_skip':
        from models.gnn_broadcast_skip import AtmosphericColumnGNN  
                
        model = AtmosphericColumnGNN(
            embed_dim=args.hidden_dim,
            depth=args.layers, 
            dropout=args.dropout,
            max_skip=args.max_skip,
            emb_dropout=args.dropout,  # Using main dropout for embedding
            channel_3d=args.channel_3d,
            channel_2d=args.channel_2d,
            channels_out=args.channel_out,
            edge_channels_in=args.edge_channels_in,
            fully_connected=args.fully_connected,
            device=device
        ).to(device)    
        
        
    elif model_name == 'gnn_edgeFeat':
        from models.gnn_edgeFeat import AtmosphericColumnGNN
        
        model = AtmosphericColumnGNN(
            embed_dim=args.hidden_dim,
            depth=args.layers, 
            dropout=args.dropout,
            max_skip=args.max_skip,
            emb_dropout=args.dropout,  # Using main dropout for embedding
            channel_3d=args.channel_3d,
            channel_2d=args.channel_2d,
            channels_out=args.channel_out,
            edge_channels_in=args.edge_channels_in,
            fully_connected=args.fully_connected,
            device=device
        ).to(device)    
        
    elif model_name == 'gnn_zeroEdge_encoding':
        from models.gnn_zeroEdge_encoding import AtmosphericColumnGNN
        
        model = AtmosphericColumnGNN(
            embed_dim=args.hidden_dim,
            depth=args.layers, 
            dropout=args.dropout,
            max_skip=args.max_skip,
            emb_dropout=args.dropout,  # Using main dropout for embedding
            channel_3d=args.channel_3d,
            channel_2d=args.channel_2d,
            channels_out=args.channel_out,
            edge_channels_in=args.edge_channels_in,
            fully_connected=args.fully_connected,
            device=device
        ).to(device)    
        
        
        
        
    elif model_name == 'rnn':
        from models.rnn import FastRnnIg
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
        
        
    else:
        raise NotImplementedError('Model has not implemented yet!')
    return model
        
    
def find_latest_checkpoint(directory):
    ckpts = glob.glob(join(directory, f'checkpoint_epoch_*.pth'))
    if len(ckpts) == 0:
        return None, None
    idx = [int(re.search( 'checkpoint_epoch_(.*?).pth', cp).group(1)) for cp in ckpts]
    return join(directory, f'checkpoint_epoch_{max(idx)}.pth'), max(idx)



class HeatingRateSmoothnessLoss(torch.nn.Module):
    def __init__(self, weight=0.1, top_levels=None):
        """
        Initialize the heating rate smoothness loss.
        
        Args:
            weight (float): Weight factor for the smoothness loss
            top_levels (int, optional): If provided, only apply smoothness loss to the top N levels.
                                      If None, apply to all levels.
        """
        super(HeatingRateSmoothnessLoss, self).__init__()
        self.weight = weight
        self.top_levels = top_levels
        
    def forward(self, outputs, x3d, x2d):
        # Calculate heating rates for all levels
        hr_pred = calculate_heating_rates(outputs, x3d, x2d)
        
        # Get the differences between adjacent levels
        hr_diff_pred = hr_pred[:, 1:, :] - hr_pred[:, :-1, :]
        
        # If top_levels is set, only use the top N levels
        if self.top_levels is not None:
            # Ensure we're not asking for more levels than available
            n_levels = min(self.top_levels, hr_diff_pred.shape[1])
            
            # Get the top N levels - in the atmospheric context, lower indices typically 
            # represent higher altitudes (top of atmosphere)
            hr_diff_pred = hr_diff_pred[:, -n_levels:, :]
            
        # Calculate mean squared differences for smoothness
        smoothness_loss = torch.mean(hr_diff_pred**2)
        
        return self.weight * smoothness_loss



def train_model(model, train_set, valid_set, normalizer):
    
    logger.info('Train started...')
    
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=args.learning_rate,
            eps=1e-8,
            weight_decay=0.01  # ridge
        )
    else:
        raise NameError('optimizer not supported.')
    
    train_loss = MeanSquaredError().to(device)
    valid_loss = MeanSquaredError().to(device)
    train_mae = MeanAbsoluteError().to(device)
    valid_mae = MeanAbsoluteError().to(device)
    
    # Initialize smoothness loss function if enabled
    use_hr_smoothness = args.hr_smoothness_weight > 0
    if use_hr_smoothness:
        top_levels_str = f", applied to top {args.hr_smoothness_top_levels} levels" if args.hr_smoothness_top_levels else ", applied to all levels"
        logger.info(f"Using heating rate smoothness loss with weight={args.hr_smoothness_weight}{top_levels_str}")
        hr_smoothness = HeatingRateSmoothnessLoss(
            weight=args.hr_smoothness_weight,
            top_levels=args.hr_smoothness_top_levels
        ).to(device)
    
    
    cp_path = None
    p_path, cp_id = find_latest_checkpoint(checkpoint_path)
    if p_path is not None:
        cp_path = p_path
        logger.info(f'Loading checkpoint: {cp_id}, {cp_path}')
        checkpoint = torch.load(cp_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        init_epoch = cp_id
        logger.info(f'Training will continue from epoch: {init_epoch}/{args.num_epoch}')
    else:
        init_epoch = 0

    epoch_number = init_epoch
    best_loss = 1e9999999 
      

    # Training loop
    for epoch in range(init_epoch, args.num_epoch):
        t1 = time.perf_counter()
        epoch_number += 1
        
        # Track smoothness losses for epochs
        smoothness_losses_train = []
        smoothness_losses_valid = []
        
        model.train(True)        
        for i, data in enumerate(train_set):
            
            t1_1 = time.perf_counter()
            
            batch_x3, batch_x2, batch_y = data
            batch_x3, batch_x2, batch_y = batch_x3.to(device), batch_x2.to(device), batch_y.to(device)
            
            batch_x3_norm, batch_x2_norm, batch_x2_orig = normalizer.normalize(batch_x3, batch_x2)

            outputs = model(batch_x3_norm, batch_x2_norm, batch_x2_orig)
            
            mse_loss = train_loss(outputs, batch_y)
            
            
            # Calculate heating rate smoothness penalty if enabled
            if use_hr_smoothness:
                smoothness_loss = hr_smoothness(outputs, batch_x3, batch_x2)
                loss = mse_loss + smoothness_loss
                
                # Track for logging
                smoothness_losses_train.append(smoothness_loss.item())
            else:
                loss = mse_loss
                smoothness_loss = 0.0
            
            
            # The MAE is a metric of prediction accuracy - it measures the average absolute difference between the predicted and true values. Its purpose is to give a clear, interpretable measure of how much the predictions deviate from the ground truth on average.
            # The smoothness loss, on the other hand, is a regularization term designed to penalize large differences between adjacent height levels in the predicted heating rates. Its goal is to encourage smoother predictions, but not necessarily more accurate ones.
            batch_mae = train_mae(outputs, batch_y)
            
            if i > 0:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                t2_1 = time.perf_counter()
                
                if i % 100 == 99:
                    if use_hr_smoothness:
                        print(f'batch {i+1}, time:{t2_1-t1_1:.3f}, loss: {loss:.4f}, '
                              f'mse: {mse_loss:.4f}, smoothness: {smoothness_loss:.4f}, '
                              f'mean_absolute_error: {batch_mae:.4f}')
                    else:
                        print(f'batch {i+1}, time:{t2_1-t1_1:.3f}, loss: {loss:.4f}, '
                              f'mean_absolute_error: {batch_mae:.4f}')
                

        # Validation step
        model.eval()
        with torch.no_grad():
            for i, v_data in enumerate(valid_set):
                
                vx3d, vx2d, v_labels = v_data
                vx3d, vx2d, v_labels = vx3d.to(device), vx2d.to(device), v_labels.to(device)
                
                # Normalize validation data
                vx3d_norm, vx2d_norm, vx2d_orig = normalizer.normalize(vx3d, vx2d)
                
                # Forward pass with normalized data
                v_outputs = model(vx3d_norm, vx2d_norm, vx2d_orig)
                
                # Calculate validation smoothness if enabled
                if use_hr_smoothness:
                    v_smoothness_loss = hr_smoothness(v_outputs, vx3d, vx2d).item()
                    smoothness_losses_valid.append(v_smoothness_loss)
                
                valid_loss.update(v_outputs, v_labels)
                valid_mae.update(v_outputs, v_labels)                

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
        if use_hr_smoothness:
            if smoothness_losses_train:
                avg_smoothness_train = sum(smoothness_losses_train) / len(smoothness_losses_train)
                log_dict['hr_smoothness'] = avg_smoothness_train

            if smoothness_losses_valid:
                avg_smoothness_valid = sum(smoothness_losses_valid) / len(smoothness_losses_valid)
                log_dict['val_hr_smoothness'] = avg_smoothness_valid
        
        # Log to wandb
        wandb.log(log_dict)

        # Print epoch summary - simplify this part
        summary = (f'{epoch_number:03}/{args.num_epoch}: '
                  f'time: {t2-t1:.3f}, '
                  f'loss: {total_train_loss:.4f}, '
                  f'mean_absolute_error: {total_train_mae:.4f}, '
                  f'val_loss: {total_valid_loss:.4f}, '
                  f'val_mean_absolute_error: {total_valid_mae:.4f}')
                  
        # Add smoothness metrics to summary if they exist
        if use_hr_smoothness and smoothness_losses_train:
            avg_smoothness_train = sum(smoothness_losses_train) / len(smoothness_losses_train)
            summary += f', hr_smoothness: {avg_smoothness_train:.4f}'
            
        if use_hr_smoothness and smoothness_losses_valid:
            avg_smoothness_valid = sum(smoothness_losses_valid) / len(smoothness_losses_valid)
            summary += f', val_hr_smoothness: {avg_smoothness_valid:.4f}'
                
                
        print(summary)
        
        # Save checkpoints
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': valid_loss,
        }
        torch.save(
            checkpoint, 
            join(checkpoint_path, f'checkpoint_epoch_{epoch_number}.pth')
        )
        if total_valid_loss < best_loss:
            torch.save(checkpoint, join(checkpoint_path, 'best_model.pth'))
            best_loss = total_valid_loss
            
        train_mae.reset()
        valid_mae.reset()
        train_loss.reset()
        valid_loss.reset()
        
    return model


def calculate_heating_rates(y, x3d, x2d): 
    # assumed output order is: [lw_up, lw_dn, sw_up, sw_dn]
    g = 9.80665
    cd = 1005
    cv = 1855
    qv_ecrad_in_idx = 5
    pres_ecrad_in_idx = 2
    pres_sfc_ecrad_in_idx = 0

    qv = x3d[..., qv_ecrad_in_idx]
    pres = torch.zeros([x3d.shape[0], 71], device=y.device)

    pres[..., 1:] = x3d[..., pres_ecrad_in_idx]
    pres[..., 0] = x2d[..., pres_sfc_ecrad_in_idx]
    pres[...,1:-1] = torch.sqrt(pres[...,1:-1] * pres[...,2:])
    top_interp = pres[...,-1] / torch.sqrt(pres[...,-2]*pres[...,-1])
    
    # pressure values are very small at the toa
    pres[...,1:-1] = torch.sqrt(pres[...,0:69] * pres[...,1:-1])
    pres[...,-1] = top_interp
    pres = torch.unsqueeze(pres, dim=-1)
    qv = torch.unsqueeze(qv, dim=-1)
    heating_rate = ((-g / (cd * (1-qv) + cv * qv)) / \
    (pres[..., :-1, :] - pres[...,1:,:])) * \
    ((y[..., :-1, [0, 2]] - y[..., :-1, [1, 3]]) - \
    (y[..., 1:, [0, 2]] - y[..., 1:, [1, 3]])) * 24*60*60
    return heating_rate


def process_timing_statistics(timing_data, model, test_path):
    """
    Process and report timing statistics from collected data.
    """
    batch_size = args.batch_size
    
    # Calculate basic statistics
    mean_time = sum(timing_data) / len(timing_data)
    std_time = (sum((t - mean_time) ** 2 for t in timing_data) / len(timing_data)) ** 0.5
    min_time = min(timing_data)
    max_time = max(timing_data)
    
    # Calculate per-sample metrics
    per_sample_time = mean_time / batch_size
    samples_per_second = batch_size / mean_time
    
    # Calculate Earth-wide metrics
    earth_cells = args.num_cells  # Number of columns in the entire Earth model
    earth_time = earth_cells * per_sample_time  # Time to process entire Earth
    earth_batches = math.ceil(earth_cells / batch_size)  # Number of batches needed
    earth_batch_time = earth_batches * mean_time  # Time to process Earth in batches
    
    gpu_info = torch.cuda.get_device_name(0)
    
    summary_text = f"""
    ==================== INFERENCE TIMING SUMMARY ====================
    Model: {args.model}
    Hardware: {gpu_info}
    Parameters: {count_parameters(model):,}

    Configuration:
    Hidden Dimension: {args.hidden_dim}
    Layers: {args.layers}
    Train Batch Size: {args.batch_size}
    Test Batch Size: {batch_size}

    Batch Performance:
    Batches measured: {len(timing_data)}
    Mean batch time: {mean_time*1000:.3f} ms/batch
    Std deviation: {std_time*1000:.3f} ms/batch
    Min/Max batch time: {min_time*1000:.3f}/{max_time*1000:.3f} ms/batch
    
    Sample Performance:
    Per sample time: {per_sample_time*1000:.3f} ms/sample
    Throughput: {samples_per_second:.1f} samples/second

    Earth-wide Performance (for {earth_cells:,} columns):
    Sequential processing time: {earth_time*1000:.2f} ms ({earth_time:.6f} sec)
    Batched processing time: {earth_batch_time*1000:.2f} ms ({earth_batch_time:.6f} sec)
    Required batches: {earth_batches}
    """
    
    summary_text += "\nStatistical Distribution (percentiles):\n"
    sorted_times = sorted(timing_data)
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        idx = int(len(sorted_times) * p / 100)
        summary_text += f"  {p}th percentile: {sorted_times[idx]*1000:.3f} ms\n"
        
    summary_text += "================================================================\n"
    
    # Print to console
    print(summary_text)
    
    # Save human-readable summary to text file
    with open(join(test_path, 'inference_timing_summary.txt'), 'w') as f:
        f.write(summary_text)
        
    return samples_per_second, per_sample_time, earth_time, earth_batch_time


def test_model(model, test_set, normalizer):
    logger.info('Test started...')    
 
    best_chkpt = join(checkpoint_path, 'best_model.pth')
    assert isfile(best_chkpt), 'Checkpoint not found, testing faild!'
    checkpoint = torch.load(best_chkpt, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])

    
    
    test_loss = MeanSquaredError().to(device)
    test_mae = MeanAbsoluteError().to(device)
    
    # Initialize HR smoothness loss if enabled
    use_hr_smoothness = args.hr_smoothness_weight > 0
    if use_hr_smoothness:
        hr_smoothness = HeatingRateSmoothnessLoss(
            weight=1.0,
            top_levels=args.hr_smoothness_top_levels
        ).to(device)  
        smoothness_losses = []

    y_true, y_pred = list(), list()
    h_true, h_pred = list(), list()
    
    # Timing configuration
    # -------------------------------
    num_timing_batches = 1000  
    warmup_batches = 10       
    timing_data = []
    
    # Perform warmup to stabilize GPU performance
    logger.info(f'Performing {warmup_batches} warmup passes...')
    warmup_iter = iter(test_set)
    for _ in range(warmup_batches):
        try:
            data = next(warmup_iter)
            batch_x3, batch_x2, _ = data
            batch_x3, batch_x2 = batch_x3.to(device), batch_x2.to(device)
            batch_x3_norm, batch_x2_norm, batch_x2_orig = normalizer.normalize(batch_x3, batch_x2)
            with torch.no_grad():
                _ = model(batch_x3_norm, batch_x2_norm, batch_x2_orig)
        except StopIteration:
            # If we run out of data, just break the loop
            break
    
    # Synchronize GPU to ensure warmup is complete
    # This makes sure all previous operations are finished before we start timing
    torch.cuda.synchronize()
    # -------------------------------   

    
    t1 = time.perf_counter(), time.process_time()
    
    for i, data in enumerate(test_set):
        
        batch_x3, batch_x2, batch_y = data
        batch_x3, batch_x2, batch_y = batch_x3.to(device), batch_x2.to(device), batch_y.to(device)
        
        # Normalize test data
        batch_x3_norm, batch_x2_norm, batch_x2_orig = normalizer.normalize(batch_x3, batch_x2)
        
        model.eval()
        with torch.no_grad():
            # Time inference for a subset of batches
            # -------------------------------   
            if i < num_timing_batches:
                # Ensure all previous GPU operations are complete
                torch.cuda.synchronize()
                
                # Time forward pass
                start = time.perf_counter()
                # -------------------------------   
                
                outputs = model(batch_x3_norm, batch_x2_norm, batch_x2_orig)
                
                # Ensure forward pass is complete before stopping timer
                # -------------------------------   
                torch.cuda.synchronize()
                
                end = time.perf_counter()
                timing_data.append(end - start)
                # -------------------------------   
            else:
                outputs = model(batch_x3_norm, batch_x2_norm, batch_x2_orig)
            
            # Track HR smoothness loss if enabled
            if use_hr_smoothness:
                smoothness_loss = hr_smoothness(outputs, batch_x3, batch_x2).item()
                smoothness_losses.append(smoothness_loss)
            
        y_true.append(batch_y.detach().cpu())
        y_pred.append(outputs.detach().cpu())
        h_true.append(calculate_heating_rates(batch_y, batch_x3, batch_x2).detach().cpu())
        h_pred.append(calculate_heating_rates(outputs, batch_x3, batch_x2).detach().cpu())

        loss = test_loss(outputs, batch_y)
        mae = test_mae(outputs, batch_y)
        if i % 1000 == 999:
            print(f'batch {i+1} loss: {loss:.4f}, '
                    f'mean_absolute_error: {mae:.4f},')

    t2 = time.perf_counter(), time.process_time()
    
    # Process timing statistics using dedicated function
    # -------------------------------   
    samples_per_second, per_sample_time, earth_time, earth_batch_time = None, None, None, None
    if timing_data:
        samples_per_second, per_sample_time, earth_time, earth_batch_time = process_timing_statistics(
            timing_data, model, test_path)
        
    # -------------------------------   
    
    y_true = torch.cat(y_true, 0)
    y_pred = torch.cat(y_pred, 0)
    h_true = torch.cat(h_true, 0)
    h_pred = torch.cat(h_pred, 0)

    total_test_loss = test_loss.compute()
    total_test_mae = test_mae.compute()

    # Report HR smoothness loss if calculated
    if use_hr_smoothness and smoothness_losses:
        avg_smoothness = sum(smoothness_losses) / len(smoothness_losses)
        print(f'Test time: {t2[0] - t1[0]:.2f} loss: {total_test_loss:.4f} '
              f'mean_absolute_error: {total_test_mae:.4f} '
              f'hr_smoothness: {avg_smoothness:.4f}')
        
        # Log to wandb
        wandb.log({"test_hr_smoothness": avg_smoothness})
    else:
        print(f'Test time: {t2[0] - t1[0]:.2f} loss: {total_test_loss:.4f} '
              f'mean_absolute_error: {total_test_mae:.4f}')
        


    with open(join(test_path, 'y_true.pickle'), 'wb') as handle:
        pickle.dump(y_true, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(join(test_path, 'y_pred.pickle'), 'wb') as handle:
        pickle.dump(y_pred, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(join(test_path, 'h_true.pickle'), 'wb') as handle:
        pickle.dump(h_true, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(join(test_path, 'h_pred.pickle'), 'wb') as handle:
        pickle.dump(h_pred, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # Return timing metrics for logging in wandb
    return samples_per_second, per_sample_time, earth_time, earth_batch_time


def get_column_data_with_disk_cache(filenames, subsample=args.subsample, shuffle=False, num_workers=0):
    icon_data = IconColumnIterableDataset(filenames, subsample=subsample, cache_dir='/tmp', shuffle=shuffle)

    # Prepare arguments for DataLoader
    dataloader_args = {
        'dataset': icon_data,
        'batch_size': args.batch_size,
        'pin_memory': True,
        'num_workers': num_workers,
    }

    if num_workers > 0:
        dataloader_args['prefetch_factor'] = args.prefetch_factor

    return DataLoader(**dataloader_args)



def main():
    logger.info('Code started...')
    
    FILENAMES = glob.glob(join(args.dataset, '*.h5'))
    time_indices = [float(re.search( r'\_time_(.*?)\.h5', f).group(1)) for f in FILENAMES]
    sorted_files = [x for _,x in sorted(zip(time_indices, FILENAMES))]
    
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Save RNG states before model initialization
    torch_rng_state = torch.get_rng_state()
    np_rng_state = np.random.get_state()
    random_rng_state = random.getstate()

    train_files = sorted_files[200:2000]
    val_files = sorted_files[:160] + sorted_files[2020:2180]
    test_files = sorted_files[2220:]
    
    # Sample dataset
    if args.percent < 1:
        train_files = prng.choice(train_files, max(1, int(args.percent*len(train_files))))
        val_files = prng.choice(val_files, max(1, int(args.percent*len(val_files))))
        test_files = prng.choice(test_files, max(1, int(args.percent*len(test_files))))

    stats_file = join(args.dataset, 'normalizer_stats_per_feat.pickle')
    mean2d, var2d, mean3d, var3d = get_normalization_params(stats_file)
    normalizer = DataNormalizer(mean2d, var2d, mean3d, var3d, device=device)
    
    # First create the model before initializing wandb
    model = get_model(args.model)
    num_params = count_parameters(model)
    
    # Initialize W&B here before anything else
    wandb.init(
        project='deepcloud-yves', 
        name=save_id, 
        id=save_id, 
        config={**wandb_config, **args.__dict__}, 
        sync_tensorboard=True, 
        save_code=True, 
        resume='allow', 
        tags=['icon grid',],    
        mode=args.wandb_mode
    )
    
    # Now watch the model after it's created
    wandb.watch(model, log_freq=100)
    
    # Summary metrics for the end
    summary_metrics = {}
    summary_metrics["num_parameters"] = num_params
    print(f"Trainable parameters: {num_params:,}")
    
    # Restore RNG states after model initialization
    torch.set_rng_state(torch_rng_state)
    np.random.set_state(np_rng_state)
    random.setstate(random_rng_state)

    if args.train:
        tr1 = time.perf_counter(), time.process_time()                        
        train_loader = get_column_data_with_disk_cache(train_files, shuffle=True)
        val_loader = get_column_data_with_disk_cache(val_files, shuffle=False, subsample=1.0)
        
        train_model(model, train_loader, val_loader, normalizer)
        
        tr2 = time.perf_counter(), time.process_time()
        train_real_time = tr2[0] - tr1[0]
        train_cpu_time = tr2[1] - tr1[1]
        print(f'Training time: Real time: {train_real_time:.2f}, CPU time: {train_cpu_time:.2f}')
        

    
    if args.test:
        #test1 = time.perf_counter(), time.process_time()
        test_loader = get_column_data_with_disk_cache(test_files, shuffle=False)
        
        # First measure inference time
        logger.info('---------------------------------------')
        logger.info('Step 1: Measuring inference time...')
        samples_per_second, per_sample_time, earth_time, earth_batch_time = test_model(model, test_loader, normalizer)
        
            
    # Log all summary metrics at the end
    wandb.log(summary_metrics, commit=True)
    
    # Finish the wandb run
    wandb.finish()
    
    logger.info('Code ended!')

if __name__ == '__main__':
    main()
    