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
# import lightning as L
import numpy as np
import matplotlib.pyplot as plt
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from torchinfo import summary

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
parser.add_argument('--zero-freq-indices', nargs='+', type=int, default=None, help='Zero frequency indices to zero out')



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
        ).to(device)
        
                
    elif model_name == 'afno_crossAttention_clean':
        from column_files.afno_column_crossAttention_clean import AFNONet
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
        ).to(device)
        
    elif model_name == 'afno_crossAttention_clean1':
        from column_files.afno_column_crossAttention_clean1 import AFNONet
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
        ).to(device)
        
    elif model_name == 'afno_crossAttention_expanded_clean':
        from column_files.afno_column_crossAttention_expanded_clean import AFNONet
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
        ).to(device)
            

    elif model_name == 'afno_easyConcat_clean':
        from column_files.afno_column_concatEasy_clean import AFNONet
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
        ).to(device)        
        
    elif model_name == 'afno_easyConcat_clean_histo':
        from column_files.afno_column_concatEasy_clean_histo import AFNONet
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
        ).to(device)     
        
        
    elif model_name == 'afno_easyConcat_clean_smooth':
        from column_files.afno_column_concatEasy_clean_smoothing import AFNONet
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
            zero_freq_indices=args.zero_freq_indices  # Pass the parameter
            
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



import matplotlib.pyplot as plt
import os
def plot_flux_histograms_pre_sigmoid(y_pred_pre_sigmoid, epoch, save_directory, log_to_wandb=True):
    """
    Plots and saves histograms for each flux value before sigmoid activation.

    Args:
        y_pred_pre_sigmoid (torch.Tensor): Tensor of shape [num_columns, channels_out]
        epoch (int): Current epoch number for labeling
        save_directory (str): Directory to save histogram images.
        log_to_wandb (bool): Whether to log histograms to Weights & Biases.
    """
    channels_out = y_pred_pre_sigmoid.shape[-1]
    assert channels_out == 4, f'Expected channels_out=4, but got {channels_out}'

    flux_labels = ['LW Up', 'LW Down', 'SW Up', 'SW Down']
    colors = ['blue', 'green', 'red', 'purple']

    for i in range(channels_out):
        # Flatten the flux_i to make it 1-dimensional
        flux_i = y_pred_pre_sigmoid[:, i].cpu().numpy()
        # flux_i = y_pred_pre_sigmoid[:, :, i].flatten().numpy()
        plt.figure(figsize=(8, 6))
        plt.hist(flux_i, bins=50, alpha=0.7, color=colors[i], edgecolor='black')
        plt.title(f'Epoch {epoch}: Histogram of {flux_labels[i]} Flux Values (Pre-Sigmoid)')
        plt.xlabel('Flux Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()

        # Ensure the directory exists before saving
        os.makedirs(save_directory, exist_ok=True)
        histogram_path = os.path.join(save_directory, f'histogram4_68{flux_labels[i].replace(" ", "_").lower()}_epoch_{epoch}.png')
        plt.savefig(histogram_path)
        plt.close()
        print(f'Histogram for {flux_labels[i]} flux saved as {histogram_path}')




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
    
    # Initialize loss and metric trackers
    train_loss = MeanSquaredError().to(device)
    valid_loss = MeanSquaredError().to(device)
    train_mae = MeanAbsoluteError().to(device)
    valid_mae = MeanAbsoluteError().to(device)
    
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
        
        # Set the model to evaluation mode
        model.eval()
        
        # Accumulators for collecting x_head values
        collected_x_head = []
        columns_collected = 0
        
    # Disable gradient computations by wrapping
        with torch.no_grad():
                
            for i, data in enumerate(train_set):
                
                t1_1 = time.perf_counter()
                
                batch_x3, batch_x2, batch_y = data
                batch_x3, batch_x2, batch_y = batch_x3.to(device), batch_x2.to(device), batch_y.to(device)

                outputs , x_head = model(batch_x3, batch_x2)
                
                # Accumulate x_head values
                # Select only the last height level (e.g., level 69)
                x_head_last = x_head[:, 68, :]  # Shape: [batch_size, channels_out]
                collected_x_head.append(x_head_last.detach().cpu())
                columns_collected += x_head_last.shape[0]
                
                # Accumulate until reaching target_num_columns
                if columns_collected >= 10000:
                    break  # Stop accumulating once the target is reached
                    
                #loss = train_loss(outputs, batch_y)
                #batch_mae = train_mae(outputs, batch_y)
                
                #if i > 0 and i % vbatch == 0:
                #    optimizer.zero_grad()
                #    loss.backward()
                #    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                #    optimizer.step()
                #    t2_1 = time.perf_counter()
                    
                if i % 10 == 9 or vbatch > 1:
                    print(f'batch {i+1}') # , time:{t2_1-t1_1:.3f}, loss: {loss:.4f}, mean_absolute_error: {batch_mae:.4f}')
            
          
          
        # Concatenate collected x_head values into a single tensor
        all_x_head = torch.cat(collected_x_head, dim=0)  # Shape: [num_columns, channels_out]

        # If collected more than 1000 columns, truncate
        if all_x_head.shape[0] > 5000:
            all_x_head = all_x_head[:5000]  


        # Save the all_x_head tensor to a file
        save_path = os.path.join(test_path, f'all_x_head68_epoch_{epoch_number}.pt')
        torch.save(all_x_head, save_path)
        print(f'all_x_head saved at {save_path}')
    
        # Plot histograms for the pre-sigmoid outputs
        plot_flux_histograms_pre_sigmoid(all_x_head, epoch_number, test_path)




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

        # Compute total losses and metrics
        total_train_loss = train_loss.compute()
        total_valid_loss = valid_loss.compute()
        total_train_mae = train_mae.compute()
        total_valid_mae = valid_mae.compute()

        t2 = time.perf_counter()

        # Log metrics to W&B
        wandb.log({
            'epoch': epoch_number, 
            'loss': total_train_loss,
            'val_loss': total_valid_loss,
            'mean_absolute_error': total_train_mae,
            'val_mean_absolute_error': total_valid_mae
            })

        # Print epoch summary
        print(f'{epoch_number:03}/{args.num_epoch}: ',
              f'time: {t2-t1:.3f}', 
              f'loss: {total_train_loss:.4f} ',
              f'mean_absolute_error: {total_train_mae:.4f}, ',
              f'val_loss: {total_valid_loss:.4f}, ',
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


def test_model(model, test_set):
    logger.info('Test started...')    
 
    # Load the best model checkpoint
    best_chkpt = join(checkpoint_path, 'best_model.pth')
    assert isfile(best_chkpt), 'Checkpoint not found, testing faild!'
    checkpoint = torch.load(best_chkpt, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize loss and metric trackers
    test_loss = MeanSquaredError().to(device)
    test_mae = MeanAbsoluteError().to(device)

    y_true, y_pred = list(), list()
    h_true, h_pred = list(), list()
    
    t1 = time.perf_counter(), time.process_time()
    
    
    for i, data in enumerate(test_set):
        
        batch_x3, batch_x2, batch_y = data
        batch_x3, batch_x2, batch_y = batch_x3.to(device), batch_x2.to(device), batch_y.to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(batch_x3, batch_x2)
            
        # Collect true and predicted values for further analysis
        y_true.append(batch_y.detach().cpu())
        y_pred.append(outputs.detach().cpu())
        h_true.append(calculate_heating_rates(batch_y, batch_x3, batch_x2).detach().cpu())
        h_pred.append(calculate_heating_rates(outputs, batch_x3, batch_x2).detach().cpu())

        # Calculate and log loss and mean absolute error
        loss = test_loss(outputs, batch_y)
        mae = test_mae(outputs, batch_y)
        # print(i+1)
        if i % 1000 == 999:
            print(f'batch {i+1} loss: {loss:.4f}, '
                    f'mean_absolute_error: {mae:.4f},')

    t2 = time.perf_counter(), time.process_time()
    
    # Concatenate all collected true and predicted values
    y_true = torch.cat(y_true, 0)
    y_pred = torch.cat(y_pred, 0)
    h_true = torch.cat(h_true, 0)
    h_pred = torch.cat(h_pred, 0)

    # Compute total test loss and mean absolute error
    total_test_loss = test_loss.compute()
    total_test_mae = test_mae.compute()

    print(f'Test time: {t2[0] - t1[0]:.2f} loss: {total_test_loss:.4f} ',
            f'mean_absolute_error: {total_test_mae:.4f}')
    
    # mean_err = torch.mean(torch.abs(y_true - y_pred), dim=0)
    # heat_err = torch.mean(torch.abs(h_true - h_pred), dim=0)

    # Save true and predicted values to files for further analysis
    with open(join(test_path, 'y_true.pickle'), 'wb') as handle:
        pickle.dump(y_true, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(join(test_path, 'y_pred.pickle'), 'wb') as handle:
        pickle.dump(y_pred, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(join(test_path, 'h_true.pickle'), 'wb') as handle:
        pickle.dump(h_true, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(join(test_path, 'h_pred.pickle'), 'wb') as handle:
        pickle.dump(h_pred, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

def test_loading_time(train_files, iter=10):
    logger.info('Test Loading time started...')
    dataset = get_data_with_disk_cache(train_files, shuffle=True)
        
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
    
    if args.test:
        test_loader = get_column_data_with_disk_cache(test_files)
        test_model(model, test_loader)
    
    logger.info('Code ended!')

if __name__ == '__main__':
    main()
    