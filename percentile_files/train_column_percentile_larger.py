import os
import re
import sys
import time
import glob
import h5py
import yaml
import fsspec
import pickle
import shutil
import logging
import tempfile
import random

import argparse
from concurrent.futures import ThreadPoolExecutor
from os.path import join, dirname, basename, normpath, isfile, exists

import wandb
import torch
# import lightning as L
import numpy as np
import matplotlib.pyplot as plt
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MeanAbsolutePercentageError
from torchinfo import summary
from torch.optim.lr_scheduler import StepLR


from scipy.stats import percentileofscore


import torch.autograd.profiler as profiler

from data_loaders import IconColumnIterableDataset


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

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import argparse
import os

parser = argparse.ArgumentParser(description='Train Transformer models.')
parser.add_argument('--model', type=str, default='vit', help='Name of the model to be trained')
parser.add_argument('--dataset', type=str, required=True, help='Path to dataset')
parser.add_argument('--save', type=str, required=True, help='Path to save the result')
parser.add_argument('--percent', type=float, default=1.0, help='Percent of data to train on')
parser.add_argument('--subsample', type=float, default=None, help='Subsampling rate')
parser.add_argument('--num-workers', type=int, default=os.cpu_count(), help='Number of workers to load data')
parser.add_argument('--prefetch-factor', type=int, default=2, help='Prefetch factor')
parser.add_argument('--wandb-mode', type=str, default='disabled', choices={'online', 'offline', 'disabled'}, help='Operating mode for W&B')
parser.add_argument('--num-cells', type=int, default=81920, help='Number of ICON cells')
parser.add_argument('--train', action=argparse.BooleanOptionalAction, default=True, help='Specify if training takes place')
parser.add_argument('--test', action=argparse.BooleanOptionalAction, default=True, help='Specify if test takes place')
parser.add_argument('--shuffle', action=argparse.BooleanOptionalAction, default=True, help='Shuffling the train dataset')
parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
parser.add_argument('--vbatch', type=int, default=1, help='Virtual batch: gradients will be applied after vbatch epoch')
parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer')
parser.add_argument('--clip', type=float, default=1.0, help='Gradient clipping')
parser.add_argument('--num-epoch', type=int, default=100, help='Number of epochs')
parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--patch-size', type=int, default=2, help='Patch size')
parser.add_argument('--vit-hidden-dim', type=int, default=256, help='Vit hidden dimension')
parser.add_argument('--vit-layers', type=int, default=4, help='Vit layers')
parser.add_argument('--vit-heads', type=int, default=6, help='Vit heads')
parser.add_argument('--vit-dropout', type=float, default=0.0, help='Vit dropout')
parser.add_argument('--afno-sparsity-threshold', type=float, default=0.01, help='Sparsity threshold for AFNO')
parser.add_argument('--hard-thresholding-fraction', type=float, default=1, help='hard thresholding fraction AFNO')
parser.add_argument('--cutoff-frequency', type=float, default=0.1, help='cutoff frequency low pass filtering in AFNO')
parser.add_argument('--collect-magnitudes', type=bool, default=False, help='collect magnitudes')



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


def get_column_data_with_disk_cache(filenames, subsample=args.subsample, shuffle=False):
    icon_data = IconColumnIterableDataset(filenames, subsample=subsample, cache_dir='/tmp')

    return DataLoader(
        icon_data, 
        batch_size=args.batch_size, 
        # shuffle=shuffle, deafult should be shuffle=none
        pin_memory=True, 
        num_workers=args.num_workers, # The number of subprocesses to use for data loading. Each worker will fetch samples from the dataset independently and in parallel.
        prefetch_factor=args.prefetch_factor #  The number of samples to prefetch in the background while the current batch is being processed.
    )

def get_normalization_params(stats_file):
    with open(stats_file, 'rb') as f:
        stats = pickle.load(f)
        return torch.tensor(stats['mean2d']).to(device), \
            torch.tensor(stats['var2d']).to(device), \
                torch.tensor(stats['mean3d']).to(device), \
                    torch.tensor(stats['var3d']).to(device)
    
def get_model(model_name, mean2d, var2d, mean3d, var3d, is_test):
    logger.info('Preparing the model...')
    if model_name == 'vit':
        from vit import ViT
        model = ViT(
            num_cells=args.num_cells,
            patch_size=args.patch_size,
            dim=args.vit_hidden_dim,
            mlp_dim=args.vit_hidden_dim,
            depth=args.vit_layers,
            heads=args.vit_heads,
            dropout=args.vit_dropout,
            mean2d=mean2d,
            var2d=var2d, 
            mean3d=mean3d, 
            var3d=var3d,
            device=device
        ).to(device)
        
    elif model_name == 'vit_column4':
        from vit_column import ViT4
        model = ViT4(
            num_cells=args.num_cells,
            patch_size=args.patch_size,
            dim=args.vit_hidden_dim,
            mlp_dim=args.vit_hidden_dim,
            depth=args.vit_layers,
            heads=args.vit_heads,
            dropout=args.vit_dropout,
            mean2d=mean2d,
            var2d=var2d, 
            mean3d=mean3d, 
            var3d=var3d,
            device=device
        ).to(device)
        
    
    # AFNO Implementation
    elif model_name == 'afno':
        from column_files.afno_column_clean import AFNONet
        model = AFNONet(
            num_cells=args.num_cells,
            patch_size=args.patch_size,
            embed_dim=args.vit_hidden_dim,
            # mlp_dim=args.vit_hidden_dim,
            depth=args.vit_layers, #num blocks
            dropout=args.vit_dropout, # used in the mlp
            mean2d=mean2d,
            var2d=var2d, 
            mean3d=mean3d, 
            var3d=var3d,
            device=device,
            
            is_test=args.test,  
            sparsity_threshold=args.afno_sparsity_threshold,  
            hard_thresholding_fraction = args.hard_thresholding_fraction,
            cutoff_frequency=args.cutoff_frequency
        ).to(device)

        
    elif model_name == 'afno_easyConcat':
        from column_files.afno_column_concatEasy import AFNONet
        model = AFNONet(
            num_cells=args.num_cells,
            patch_size=args.patch_size,
            embed_dim=args.vit_hidden_dim,
            depth=args.vit_layers, #num blocks
            dropout=args.vit_dropout, # used in the mlp
            mean2d=mean2d,
            var2d=var2d, 
            mean3d=mean3d, 
            var3d=var3d,
            device=device,
            
            is_test=args.test,  
            sparsity_threshold=args.afno_sparsity_threshold,  
            hard_thresholding_fraction = args.hard_thresholding_fraction,
            cutoff_frequency=args.cutoff_frequency
        ).to(device)
        
        

        
    elif model_name == 'afno_column_concatEasy_percentile':
        from afno_column_concatEasy_percentile import AFNONet
        model = AFNONet(
            num_cells=args.num_cells,
            patch_size=args.patch_size,
            embed_dim=args.vit_hidden_dim,
            depth=args.vit_layers, #num blocks
            dropout=args.vit_dropout, # used in the mlp
            mean2d=mean2d,
            var2d=var2d, 
            mean3d=mean3d, 
            var3d=var3d,
            device=device,
            
            is_test=args.test,  
            sparsity_threshold=args.afno_sparsity_threshold,  
            hard_thresholding_fraction = args.hard_thresholding_fraction,
            cutoff_frequency=args.cutoff_frequency
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

# metrics
def get_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm



def train_model(model, train_set, valid_set):
    
    # Log the start of training
    logger.info('Train started...')
    
    # Initialize Weights & Biases 
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
    wandb.watch(model, log_freq=100)

    # Set up the optimizer based on the specified type
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=args.learning_rate,
            eps=1e-8,
            weight_decay=0.01  # basically applying ridge regression L2
        )
    else:
        raise NameError('optimizer not supported.')
    
    # Initialize the learning rate scheduler
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Initialize loss and metric trackers
    train_loss = MeanSquaredError().to(device)
    valid_loss = MeanSquaredError().to(device)
    train_mae = MeanAbsoluteError().to(device)
    valid_mae = MeanAbsoluteError().to(device)
    train_mape = MeanAbsolutePercentageError().to(device)
    valid_mape = MeanAbsolutePercentageError().to(device)    
    
    # Load the latest checkpoint if available
    cp_path = None
    p_path, cp_id = find_latest_checkpoint(checkpoint_path)
    if p_path is not None:
        cp_path = p_path
        logger.info(f'Loading checkpoint: {cp_id}, {cp_path}')
        checkpoint = torch.load(cp_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        init_epoch = cp_id
        # vbatch = min(args.vbatch + (init_epoch/2), 20)
        logger.info(f'Training will continue from epoch: {init_epoch}/{args.num_epoch}')
    else:
        init_epoch = 0
        # vbatch = args.vbatch

    vbatch = args.vbatch
    epoch_number = init_epoch
    best_loss = 1e9999999 
      

    # Training loop
    for epoch in range(init_epoch, args.num_epoch):
        t1 = time.perf_counter()
        epoch_number += 1
        
        # Training step
        model.train(True)        
        for i, data in enumerate(train_set):
            
            t1_1 = time.perf_counter()
            
            batch_x3, batch_x2, batch_y = data
            batch_x3, batch_x2, batch_y = batch_x3.to(device), batch_x2.to(device), batch_y.to(device)

            outputs = model(batch_x3, batch_x2)
            loss = train_loss(outputs, batch_y)
            batch_mae = train_mae(outputs, batch_y)
            batch_mape = train_mape(outputs, batch_y)

            
            if i > 0 and i % vbatch == 0:
                optimizer.zero_grad()
                loss.backward()
                
                # Compute gradient norm
                gradient_norm = get_gradient_norm(model)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                t2_1 = time.perf_counter()
                
                if i % 100 == 99 or vbatch > 1:
                    print(f'batch {i+1}, time:{t2_1-t1_1:.3f}, loss: {loss:.4f}, mean_absolute_error: {batch_mae:.4f}')
                

        # Validation step
        model.eval()
        

        with torch.no_grad():
            for i, v_data in enumerate(valid_set):
                
                vx3d, vx2d, v_labels = v_data
                vx3d, vx2d, v_labels = vx3d.to(device), vx2d.to(device), v_labels.to(device)
                v_outputs = model(vx3d, vx2d)        
                
                # two different losses         
                valid_loss.update(v_outputs, v_labels)
                valid_mae.update(v_outputs, v_labels)   
                valid_mape.update(v_outputs, v_labels)


        # Compute total losses and metrics
        total_train_loss = train_loss.compute()
        total_valid_loss = valid_loss.compute()
        total_train_mae = train_mae.compute()
        total_valid_mae = valid_mae.compute()
        total_train_mape = train_mape.compute()
        total_valid_mape = valid_mape.compute()
        
        t2 = time.perf_counter()
        
        # Update the learning rate scheduler
        scheduler.step(total_valid_loss)

        # Log metrics to W&B
        wandb.log({
            'gradient_norm': gradient_norm, # added gradient norm
            'epoch': epoch_number, 
            'loss': total_train_loss,
            'val_loss': total_valid_loss,
            'mean_absolute_error': total_train_mae,
            'mean_absolute_percentage_error': total_train_mape,
            'val_mean_absolute_error': total_valid_mae,
            'learning_rate': optimizer.param_groups[0]['lr']  # Log the learning rate

            })

        # Print epoch summary
        print(f'{epoch_number:03}/{args.num_epoch}: ',
              f'time: {t2-t1:.3f}', 
              f'loss: {total_train_loss:.4f} ',
              f'mean_absolute_error: {total_train_mae:.4f}, ',
              f'val_loss: {total_valid_loss:.4f}, ',
              f'mean_absolute_percentage_error: {total_train_mape:.4f}, ',
              f'val_mean_absolute_error: {total_valid_mae:.4f}')
        
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
            
        # Reset metrics for the next epoch
        train_mae.reset()
        valid_mae.reset()
        train_loss.reset()
        valid_loss.reset()
        valid_mape.reset()

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





import matplotlib.pyplot as plt
import os

def plot_sorted_magnitudes(all_magnitudes_sorted, lambda_5, lambda_10, lambda_20, lambda_30, test_path):
    """
    Plot the sorted Fourier magnitudes and visualize 5%, 10%, 20%, and 30% quantiles.
    Save the plot to the test path.
    """
    # Plot the sorted Fourier magnitudes
    plt.figure(figsize=(14, 8))
    plt.plot(all_magnitudes_sorted, label="Sorted Fourier Magnitudes", color="blue", alpha=0.7)

    # Add horizontal lines for the quantiles with improved colors and styles
    plt.axhline(y=lambda_5, color="red", linestyle="--", linewidth=2, label=f"5% Quantile: {lambda_5:.6f}")
    plt.axhline(y=lambda_10, color="green", linestyle="--", linewidth=2, label=f"10% Quantile: {lambda_10:.6f}")
    plt.axhline(y=lambda_20, color="orange", linestyle="--", linewidth=2, label=f"20% Quantile: {lambda_20:.6f}")
    plt.axhline(y=lambda_30, color="purple", linestyle="--", linewidth=2, label=f"30% Quantile: {lambda_30:.6f}")

    # Add labels and title
    plt.title("Sorted Fourier Magnitudes with Sparsification Quantile Cutoffs", fontsize=16)
    plt.xlabel("Index (Fourier Components)", fontsize=14)
    plt.ylabel("Magnitude", fontsize=14)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3)

    # Save the plot to the test_path directory
    plot_file = os.path.join(test_path, 'quantile_plot_full_sparsification.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Full sparsification plot saved to {plot_file}")
    plt.close()
    
def plot_sorted_magnitudes_zoom(all_magnitudes_sorted, lambda_values, test_path, specified_values_quantiles):
    """
    Create a zoomed-in view focusing specifically on higher quantile ranges for sparsification.
    Shows the upper portion of the magnitude distribution where sparsification decisions matter most.
    """
    # Unpack lambda values for higher percentiles
    lambda_40 = lambda_values['lambda_40']
    lambda_50 = lambda_values['lambda_50']
    lambda_60 = lambda_values['lambda_60']
    lambda_70 = lambda_values['lambda_70']
    lambda_80 = lambda_values['lambda_80']
    
    # Create a focused plot on the upper range where higher quantiles matter
    plt.figure(figsize=(14, 8))
    
    # Focus on the upper 30% of components where the meaningful variation occurs
    upper_start_idx = int(len(all_magnitudes_sorted) * 0.7)  # Start from 70th percentile
    x_upper = np.arange(upper_start_idx, len(all_magnitudes_sorted))
    y_upper = all_magnitudes_sorted[upper_start_idx:]
    
    # Plot the upper range with better visibility
    plt.plot(x_upper, y_upper, color="blue", alpha=0.8, linewidth=1.5, label="Sorted Fourier Magnitudes")
    
    # Add horizontal lines for higher quantiles with distinct colors
    plt.axhline(y=lambda_40, color="red", linestyle="--", linewidth=2, label=f"40% Quantile: {lambda_40:.4f}")
    plt.axhline(y=lambda_50, color="green", linestyle="--", linewidth=2, label=f"50% Quantile: {lambda_50:.4f}")
    plt.axhline(y=lambda_60, color="orange", linestyle="--", linewidth=2, label=f"60% Quantile: {lambda_60:.4f}")
    plt.axhline(y=lambda_70, color="purple", linestyle="--", linewidth=2, label=f"70% Quantile: {lambda_70:.4f}")
    plt.axhline(y=lambda_80, color="brown", linestyle="--", linewidth=2, label=f"80% Quantile: {lambda_80:.4f}")
    
    # Add vertical lines to show percentile positions
    percentile_positions = [
        int(len(all_magnitudes_sorted) * 0.4),   # 40th percentile position
        int(len(all_magnitudes_sorted) * 0.5),   # 50th percentile position
        int(len(all_magnitudes_sorted) * 0.6),   # 60th percentile position
        int(len(all_magnitudes_sorted) * 0.7),   # 70th percentile position
        int(len(all_magnitudes_sorted) * 0.8),   # 80th percentile position
    ]
    
    for i, pos in enumerate(percentile_positions):
        if pos >= upper_start_idx:  # Only show if within our zoomed range
            percentile = [40, 50, 60, 70, 80][i]
            plt.axvline(x=pos, color="gray", linestyle=":", alpha=0.6, linewidth=1)
            plt.text(pos, max(y_upper) * 0.9, f"{percentile}%", rotation=90, 
                    verticalalignment='top', horizontalalignment='right', fontsize=10)
    
    # Add a statistics box
    stats_text = f"Total components: {len(all_magnitudes_sorted):,}\n"
    stats_text += f"Showing upper 30% ({len(y_upper):,} components)\n"
    stats_text += f"Max magnitude: {max(all_magnitudes_sorted):.4f}\n"
    stats_text += f"Range: {min(y_upper):.4f} - {max(y_upper):.4f}"
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgray", alpha=0.8),
             verticalalignment='top', fontsize=11)
    
    # Formatting
    plt.title("Sparsification Range - Upper Quantiles (40%-80%)", fontsize=16, fontweight='bold')
    plt.xlabel("Index (Fourier Components)", fontsize=14)
    plt.ylabel("Magnitude", fontsize=14)
    plt.legend(fontsize=11, loc='upper left', bbox_to_anchor=(0.7, 1))
    plt.grid(True, alpha=0.3)
    
    # Use scientific notation for y-axis if values are very small
    if max(y_upper) < 0.01:
        plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    
    # Save the plot
    zoomed_plot_file = os.path.join(test_path, 'quantile_plot_high_percentiles_zoom.png')
    plt.savefig(zoomed_plot_file, dpi=300, bbox_inches='tight')
    print(f"High percentiles zoomed plot saved to {zoomed_plot_file}")
    plt.close()





def test_model(model, test_set):
    logger.info('Test started...')    
 
    # Load the best model checkpoint
    best_chkpt = join(checkpoint_path, 'best_model.pth')
    assert isfile(best_chkpt), 'Checkpoint not found, testing failed!'
    checkpoint = torch.load(best_chkpt, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Reset and enable magnitude collection
    model.reset_magnitude_collection()
    model.enable_magnitude_collection(True)
    
    t1 = time.perf_counter(), time.process_time()
    
    # Initialize the counter and set maximum samples
    total_samples = 0
    max_samples = 1000
    
    base_filename = join(test_path, 'all_magnitudes_test')
    file_path = f"{base_filename}.npy"   # For the .npy file
    plot_file = f"{base_filename}.png"   # For the plot file
    pickle_file = f"{base_filename}_percentiles.pickle"  # For the pickle file

    # Check if the magnitudes file already exists
    if os.path.exists(file_path):
        all_magnitudes = np.load(file_path)
        print(f"Loaded magnitudes from saved file: {file_path}")
    
    else: 
        # Perform inference and collect magnitudes
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_set):
                batch_x3, batch_x2, batch_y = data
                batch_x3, batch_x2, batch_y = batch_x3.to(device), batch_x2.to(device), batch_y.to(device)
                batch_size = batch_x3.size(0)  # Get the number of samples in the current batch
                
                # Forward pass collects magnitudes automatically if enabled
                outputs = model(batch_x3, batch_x2)
                    
                # Update the total number of samples processed
                total_samples += batch_size
                if total_samples % 100 == 0:
                    print(f'Processed {total_samples} samples')
                
                # Break the loop if we've reached the maximum number of samples
                if total_samples >= max_samples:
                    break   
        
        t2 = time.perf_counter(), time.process_time()
        print(f'Processing time: Real time: {t2[0] - t1[0]:.2f}s, CPU time: {t2[1] - t1[1]:.2f}s')
        
        # Process collected magnitudes
        if not model.collected_magnitudes:
            logger.error("No magnitudes were collected. Ensure the model is correctly configured.")
            return
            
        # Concatenate all magnitudes and flatten them
        all_magnitudes = torch.cat([mag.reshape(-1) for mag in model.collected_magnitudes], dim=0)
        all_magnitudes = all_magnitudes.cpu().numpy()
        
        # Save the magnitudes
        np.save(file_path, all_magnitudes)
        print(f"Saved {len(all_magnitudes)} magnitude values to {file_path}")
        
    # Sort the magnitudes in ascending order for percentile calculations
    all_magnitudes_sorted = np.sort(all_magnitudes)
            
    # Compute quantiles to determine thresholds - focusing on higher percentiles for sparsification
    lambda_40 = np.percentile(all_magnitudes_sorted, 40)
    lambda_50 = np.percentile(all_magnitudes_sorted, 50)
    lambda_60 = np.percentile(all_magnitudes_sorted, 60)
    lambda_70 = np.percentile(all_magnitudes_sorted, 70)
    lambda_80 = np.percentile(all_magnitudes_sorted, 80)

    logger.info(f'Lambda 40th percentile (removing bottom 40%): {lambda_40}')
    logger.info(f'Lambda 50th percentile (removing bottom 50%): {lambda_50}')
    logger.info(f'Lambda 60th percentile (removing bottom 60%): {lambda_60}')
    logger.info(f'Lambda 70th percentile (removing bottom 70%): {lambda_70}')
    logger.info(f'Lambda 80th percentile (removing bottom 80%): {lambda_80}')

    # Now you can use these lambda values in your thresholding step
    print(f"Lambda 40% (Removing bottom 40% of components): {lambda_40}")
    print(f"Lambda 50% (Removing bottom 50% of components): {lambda_50}")
    print(f"Lambda 60% (Removing bottom 60% of components): {lambda_60}")
    print(f"Lambda 70% (Removing bottom 70% of components): {lambda_70}")
    print(f"Lambda 80% (Removing bottom 80% of components): {lambda_80}")

    # Save lambda parameters to a file
    lambda_values = {
        'lambda_40': float(lambda_40),
        'lambda_50': float(lambda_50),
        'lambda_60': float(lambda_60),
        'lambda_70': float(lambda_70),
        'lambda_80': float(lambda_80),
    }

    with open(pickle_file, 'wb') as handle:
        pickle.dump(lambda_values, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Lambda values saved to {pickle_file}")

    # Calculate quantiles for specific values of interest (more relevant for sparsification)
    specified_values = [0.05, 0.1, 0.15, 0.2, 0.25]
    specified_values_quantiles = {}
    for v in specified_values:
        quantile = percentileofscore(all_magnitudes_sorted, v, kind='weak') / 100.0
        specified_values_quantiles[v] = quantile
        print(f"Value {v} corresponds to quantile {quantile:.4f} ({quantile:.2%})")

    # Create zoomed-in plot focusing on higher quantiles
    plot_sorted_magnitudes_zoom(all_magnitudes_sorted, lambda_values, test_path, specified_values_quantiles)
    
    # Disable magnitude collection to save memory in future runs
    model.enable_magnitude_collection(False)
    
    return lambda_values


def test_loading_time(train_files, iter=10):
    logger.info('Test Loading time started...')
    dataset = get_column_data_with_disk_cache(train_files, shuffle=True)
        
    for it in range(iter):
        t1 = time.perf_counter(), time.process_time()
        for i, data in enumerate(dataset):
            batch_x3, batch_x2, batch_y = data
            batch_x3, batch_x2, batch_y = batch_x3.to(device), batch_x2.to(device), batch_y.to(device)
        t2 = time.perf_counter(), time.process_time()
        logger.info(f'Iteration: {it}: Real time: {t2[0] - t1[0]:.2f}, CPU time: {t2[1]-t1[1]}')
        import gc
        gc.collect()


def main():
    logger.info('Code started...')
    
    FILENAMES = glob.glob(join(args.dataset, '*.h5'))
    time_indices = [float(re.search( r'\_time_(.*?)\.h5', f).group(1)) for f in FILENAMES]
    sorted_files = [x for _,x in sorted(zip(time_indices, FILENAMES))]

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
    model = get_model(args.model, mean2d, var2d, mean3d, var3d, args.test)
    
    num_params = count_parameters(model)
    print(f"The model has {num_params:,} trainable parameters.")

    if args.train:
        tr1 = time.perf_counter(), time.process_time()                        

        train_loader = get_column_data_with_disk_cache(train_files, shuffle=args.shuffle)
        val_loader = get_column_data_with_disk_cache(val_files, subsample=1.0)
   
        
        train_model(model, train_loader, val_loader)

        tr2 = time.perf_counter(), time.process_time()
        print(f'Training time: Real time: {tr2[0] - tr1[0]:.2f}, CPU time: {tr2[1]-tr1[1]}')
    
    if args.test: # here put in train file and train loader as I evulate the percentiles
        train_loader = get_column_data_with_disk_cache(train_files)
        test_model(model, train_loader)
    
    logger.info('Code ended!')

if __name__ == '__main__':
    main()
    
