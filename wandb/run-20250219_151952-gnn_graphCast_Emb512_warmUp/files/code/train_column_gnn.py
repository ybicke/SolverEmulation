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
import math

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


from data_loaders_new import IconColumnIterableDataset
from flux_specific_sigmoid.FluxSpecificSigmoid_lwdown import load_gaussian_parameters, construct_gaussian_params_by_height



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
parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer')
parser.add_argument('--clip', type=float, default=1.0, help='Gradient clipping')
parser.add_argument('--num-epoch', type=int, default=100, help='Number of epochs')
parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--hidden-dim', type=int, default=256, help='hidden dimension')
parser.add_argument('--layers', type=int, default=4, help='layers')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
parser.add_argument('--lr-schedule-type', type=str, default='none', choices=['none', 'plateau', 'graphcast'], help='LR schedule mode to use')
parser.add_argument('--warmup-steps', type=int, default=1000, help='Warmup steps for GraphCast')
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
        

    
    elif model_name == 'gnn_mp_column':
        from column_files.gnn_mp_column import GNNColumnNet
        model = GNNColumnNet(
            num_cells=args.num_cells,
            embed_dim=args.hidden_dim,
            depth=args.layers, #num blocks
            dropout=args.dropout, # used in the mlp
            mean2d=mean2d,
            var2d=var2d, 
            mean3d=mean3d, 
            var3d=var3d,
            device=device,
            is_test=args.test,  
        
            
        ).to(device)  
        
        
    elif model_name == 'gnn_gat_column':
        from column_files.gnn_gat_column import GNNColumnNet    
        model = GNNColumnNet(
            num_cells=args.num_cells,
            embed_dim=args.hidden_dim,
            depth=args.layers, #num blocks
            dropout=args.dropout, # used in the mlp
            mean2d=mean2d,
            var2d=var2d, 
            mean3d=mean3d, 
            var3d=var3d,
            device=device,
            is_test=args.test,  
            heads=args.heads
                        
            
        ).to(device)   
        
    elif model_name == 'gnn_graphCast_column':
        from column_files.gnn_graphCast import AtmosphericColumnGNN    
        model = AtmosphericColumnGNN(
            num_cells=args.num_cells,
            embed_dim=args.hidden_dim,
            depth=args.layers, #num blocks
            dropout=args.dropout, # used in the mlp
            mean2d=mean2d,
            var2d=var2d, 
            mean3d=mean3d, 
            var3d=var3d,
            device=device,
            is_test=args.test,  
        ).to(device)  
           
    else:
        raise NotImplementedError('Model has not implemented yet!')

    # Example custom weight initialization after moving model to device
    for name, param in model.named_parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param, gain=0.01)

    return model
        
    
def find_latest_checkpoint(directory):
    ckpts = glob.glob(join(directory, f'checkpoint_epoch_*.pth'))
    if len(ckpts) == 0:
        return None, None
    idx = [int(re.search( 'checkpoint_epoch_(.*?).pth', cp).group(1)) for cp in ckpts]
    return join(directory, f'checkpoint_epoch_{max(idx)}.pth'), max(idx)


def train_model(model, train_set, valid_set, steps_per_epoch):
    logger.info('Train started...')

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

    # Set up the optimizer
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,  # Will be max_lr when using GraphCast schedule
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
    else:
        raise NameError('optimizer not supported.')
    
    #####################################################################
    # 1) Determine which scheduler to create based on args.lr_schedule_type
    #####################################################################
    scheduler_mode = args.lr_schedule_type  # e.g. 'none', 'plateau', or 'graphcast'
    scheduler = None
    
    if scheduler_mode == 'plateau':
        # A standard PyTorch scheduler that reduces LR on plateau in validation loss
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.2,
            patience=5
        )
        logger.info('Using ReduceLROnPlateau scheduler.')
    
    elif scheduler_mode == 'graphcast':
        # We will manually implement warmup + half-cosine in the loop below
        logger.info('Using GraphCast two-phase (warmup + half-cosine) scheduling.')
    
    else:
        logger.info('No scheduler. Using constant LR = args.learning_rate')
    
    #####################################################################
    # 2) Initialize losses, metrics, checkpoint, etc.
    #####################################################################
    train_loss = MeanSquaredError().to(device)
    valid_loss = MeanSquaredError().to(device)
    train_mae = MeanAbsoluteError().to(device)
    valid_mae = MeanAbsoluteError().to(device)

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
    best_loss = float('inf') 

    # For GraphCast scheduler:
    warmup_steps = args.warmup_steps
    total_optimization_steps = args.num_epoch * steps_per_epoch
    print(f'total_optimization_steps: {total_optimization_steps}')
    global_step = 0

    #####################################################################
    # 3) Main training loop
    #####################################################################
    for epoch in range(init_epoch, args.num_epoch):
        t1 = time.perf_counter()
        epoch_number += 1
        
        # TRAIN
        model.train(True)
        for i, data in enumerate(train_set):
            batch_x3, batch_x2, batch_y = data
            batch_x3, batch_x2, batch_y = (
                batch_x3.to(device), 
                batch_x2.to(device), 
                batch_y.to(device)
            )
            outputs = model(batch_x3, batch_x2)

            loss = train_loss(outputs, batch_y)
            batch_mae = train_mae(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            # If we're using the GraphCast schedule, we manually adjust LR after each batchstep
            if scheduler_mode == 'graphcast':
                global_step += 1
                if global_step <= warmup_steps:
                    # Phase 1: linear warmup 0 -> max LR
                    lr = args.learning_rate * (global_step / warmup_steps)
                else:
                    # Phase 2: half-cosine decay from max LR -> 0
                    # local_step = steps into the decay phase
                    local_step = global_step - warmup_steps
                    decay_steps = max(1, total_optimization_steps - warmup_steps)
                    lr_scale = 0.5 * (1.0 + math.cos(math.pi * local_step / decay_steps))
                    lr = args.learning_rate * lr_scale
                
                # Update the LR for the optimizer
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            # Optional: print info
            if i % 100 == 0:
                curr_lr = optimizer.param_groups[0]['lr']
                logger.info(f'Epoch={epoch_number}, batch={i}, LR={curr_lr:.6f}, loss={loss.item():.4f}, mae={batch_mae.item():.4f}')

        # VALIDATION
        model.eval()
        with torch.no_grad():
            for v_data in valid_set:
                vx3d, vx2d, v_labels = v_data
                vx3d, vx2d, v_labels = (
                    vx3d.to(device),
                    vx2d.to(device),
                    v_labels.to(device)
                )
                v_outputs = model(vx3d, vx2d)
                
                valid_loss.update(v_outputs, v_labels)
                valid_mae.update(v_outputs, v_labels)

        # Compute final epoch losses
        total_train_loss = train_loss.compute()
        total_valid_loss = valid_loss.compute()
        total_train_mae = train_mae.compute()
        total_valid_mae = valid_mae.compute()
        t2 = time.perf_counter()

        # Step the standard scheduler if using 'plateau'
        if scheduler_mode == 'plateau':
            scheduler.step(total_valid_loss)
            curr_lr = optimizer.param_groups[0]['lr']
            wandb.log({'learning_rate': curr_lr})

        # Otherwise, if using 'graphcast', we've already updated per batch,
        # so we just log the last lr from the final param_group
        elif scheduler_mode == 'graphcast':
            curr_lr = optimizer.param_groups[0]['lr']
        else:
            curr_lr = optimizer.param_groups[0]['lr']  # constant LR

        # Log epoch metrics
        wandb.log({
            'epoch': epoch_number, 
            'loss': total_train_loss,
            'val_loss': total_valid_loss,
            'mean_absolute_error': total_train_mae,
            'val_mean_absolute_error': total_valid_mae,
            'learning_rate': curr_lr
        })

        logger.info(
            f'{epoch_number:03}/{args.num_epoch} | '
            f'Time {t2 - t1:.3f}s | '
            f'LR {curr_lr:.6f} | '
            f'loss {total_train_loss:.4f} | '
            f'mae {total_train_mae:.4f} | '
            f'val_loss {total_valid_loss:.4f} | '
            f'val_mae {total_valid_mae:.4f}'
        )

        # Save checkpoints
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': valid_loss,
        }
        torch.save(checkpoint, join(checkpoint_path, f'checkpoint_epoch_{epoch_number}.pth'))
        if total_valid_loss < best_loss:
            torch.save(checkpoint, join(checkpoint_path, 'best_model.pth'))
            best_loss = total_valid_loss

        # Reset metrics for next epoch
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
        
        
def estimate_num_columns(train_files, subsample):
    columns_per_file = 81920
    actual_file_count = len(train_files)
    total_cols_subsampled = int(columns_per_file * subsample)
    approx_total_cols = actual_file_count * total_cols_subsampled
    return approx_total_cols

        
        
def get_column_data_with_disk_cache(filenames, subsample=args.subsample, shuffle=False, num_workers=0):
    icon_data = IconColumnIterableDataset(filenames, subsample=subsample, cache_dir='/tmp', shuffle=shuffle)

    # Prepare arguments for DataLoader
    dataloader_args = {
        'dataset': icon_data,
        'batch_size': args.batch_size,
        'pin_memory': True,
        'num_workers': num_workers,
    }

    # Include prefetch_factor only if num_workers > 0
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
    model = get_model(args.model, mean2d, var2d, mean3d, var3d, args.test)
    num_params = count_parameters(model)
    print(f"The model has {num_params:,} trainable parameters.")
    
    # Restore RNG states after model initialization
    torch.set_rng_state(torch_rng_state)
    np.random.set_state(np_rng_state)
    random.setstate(random_rng_state)

    if args.train:
        tr1 = time.perf_counter(), time.process_time()                        

        train_loader = get_column_data_with_disk_cache(train_files, shuffle=True)
        val_loader = get_column_data_with_disk_cache(train_files, shuffle = False, subsample=1.0)

        
        # Logic to estimate the number of columns and steps per epoch for GraphCast scheduler
        # Warmup steps is sugesteted to be 1-2 epochs and depends on the batch size 
        total_columns = estimate_num_columns(train_files, args.subsample)
        steps_per_epoch = total_columns // args.batch_size
        if total_columns % args.batch_size != 0:
            steps_per_epoch += 1
        logger.info(f"Estimated {total_columns} total columns with subsample={args.subsample}. "
                    f"Steps per epoch {steps_per_epoch} at batch_size={args.batch_size}.")
        
        
        train_model(model, train_loader, val_loader, steps_per_epoch)

        tr2 = time.perf_counter(), time.process_time()
        print(f'Training time: Real time: {tr2[0] - tr1[0]:.2f}, CPU time: {tr2[1]-tr1[1]}')
    
    if args.test:
        test_loader = get_column_data_with_disk_cache(test_files, shuffle=False)
        test_model(model, test_loader)
    
    logger.info('Code ended!')

if __name__ == '__main__':
    main()
    
