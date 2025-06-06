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

from data_loader_3d_tendency import IconIterableDataset3DTendency
from data_utils import DataNormalizer, interpolate_w_to_full_levels

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

parser = argparse.ArgumentParser(description='Train 3D GNN Tendency models.')

# General parameters
parser.add_argument('--dataset-input', type=str, required=True, help='Path to input dataset')
parser.add_argument('--dataset-output', type=str, required=True, help='Path to output dataset')
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

# GNN 3d specific parameters
parser.add_argument('--embed-dim', type=int, default=256, help='Embedding dimension for GNN')
parser.add_argument('--layers', type=int, default=4, help='Number of layers for GNN')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
parser.add_argument('--channel-3d', type=int, default=13, help='Number of 3D input channels')
parser.add_argument('--channel-2d', type=int, default=3, help='Number of 2D input channels')
parser.add_argument('--channels-out', type=int, default=7, help='Number of output channels')
parser.add_argument('--edge-channels-in', type=int, default=1, help='Number of edge feature channels')
parser.add_argument('--grid-file-path', type=str, required=True, help='Path to the ICON grid file')
parser.add_argument('--triangle-id', type=int, default=1, help='ID of the triangle to use')
parser.add_argument('--height-in', type=int, default=70, help='Number of height levels')

parser.add_argument('--triangle-division-factor', type=int, default=4, 
                    help='Factor by which to divide the triangle (1 for full 4096 columns, 4 for 1024 columns, 16 for 256 columns)')
parser.add_argument('--fully-connected', action=argparse.BooleanOptionalAction, default=False, help='Use fully connected graph')
parser.add_argument('--disable-horizontal', action=argparse.BooleanOptionalAction, default=False, help='Disable horizontal edges')

parser.add_argument('--model', type=str, default='gnn_3d_tendency', 
                    help='Model type to use for training')

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
        return torch.tensor(stats['mean2d'], dtype=torch.float32).to(device), \
            torch.tensor(stats['var2d'], dtype=torch.float32).to(device), \
            torch.tensor(stats['mean3d'], dtype=torch.float32).to(device), \
            torch.tensor(stats['var3d'], dtype=torch.float32).to(device)

def get_model():
    logger.info('Preparing the model...')
    
    from gnn_3d_tendency import GNN3dTendency   
    
    model = GNN3dTendency(
        total_cols=args.num_cells,
        grid_file_path=args.grid_file_path,
        triangle_id=args.triangle_id,
        embed_dim=args.embed_dim,
        depth=args.layers,
        dropout=args.dropout,
        channels_in_3d=args.channel_3d,
        channels_in_2d=args.channel_2d,
        channels_out=args.channels_out,
        edge_channels_in=args.edge_channels_in,
        num_height_levels=args.height_in,
        device=device,
        division_factor=args.triangle_division_factor,
        fully_connected=args.fully_connected,
        disable_horizontal=args.disable_horizontal
    ).to(device)
        
    return model

def find_latest_checkpoint(directory):
    ckpts = glob.glob(join(directory, f'checkpoint_epoch_*.pth'))
    if len(ckpts) == 0:
        return None, None
    idx = [int(re.search( 'checkpoint_epoch_(.*?).pth', cp).group(1)) for cp in ckpts]
    return join(directory, f'checkpoint_epoch_{max(idx)}.pth'), max(idx)


def transform_targets(batch_y, means, variances, k=4, min_scale=1e-6):
    """Standardize targets to zero-mean, unit-variance representation.
    
    Args:
        batch_y: Raw targets (B,N,L,C) in physical units
        means, variances: Per-channel statistics (C,)
        k: Scale factor for normalizing to approx. [-1,1] range
        min_scale: Lower bound for scaling to avoid div-by-zero
    """
    scale = torch.clamp(k * torch.sqrt(variances), min=min_scale)
    means = means.view(1, 1, 1, -1).expand_as(batch_y)
    scale = scale.view(1, 1, 1, -1).expand_as(batch_y)
    
    return (batch_y - means) / scale


def inverse_transform_targets(y_norm, means, variances, k=4, min_scale=1e-6):
    """Convert normalized values back to physical units."""
    scale = torch.clamp(k * torch.sqrt(variances), min=min_scale)
    means = means.view(1, 1, 1, -1).expand_as(y_norm)
    scale = scale.view(1, 1, 1, -1).expand_as(y_norm)
    
    return y_norm * scale + means


def precompute_train_target_mean(train_set):
    """
    Compute mean of target variables from training data for baseline comparison.
    This serves as a simple 'predict-the-mean' baseline model.
    """
    logger.info('Computing target statistics from training data...')
    train_targets = []
    for _, _, batch_y, _ in train_set:
        train_targets.append(batch_y.cpu())
    train_targets = torch.cat(train_targets, dim=0)
    train_target_mean = torch.mean(train_targets, dim=0)  # Compute mean along batch dimension
    return train_target_mean


def train_model(model, train_set, valid_set, normalizer, target_means, target_vars):
    
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
        tags=['icon grid', 'tendency', '3d'],    
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
            weight_decay=0.01  # L2 regularization
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
        logger.info(f'Training will continue from epoch: {init_epoch}/{args.num_epoch}')
    else:
        init_epoch = 0

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
            
            batch_x3, batch_x2, batch_y, batch_w = data
            batch_x3, batch_x2, batch_y, batch_w = batch_x3.to(device), batch_x2.to(device), batch_y.to(device), batch_w.to(device)
            

            
            # Interpolate vertical wind and concatenate to x3d
            w_full = interpolate_w_to_full_levels(batch_w)
            batch_x3_with_w = torch.cat([batch_x3, w_full], dim=-1)
            
            # Normalize the inputs using the DataNormalizer
            batch_x3_norm_with_w, batch_x2_norm, _ = normalizer.normalize(batch_x3_with_w, batch_x2)
            
            # Transform targets
            batch_y_transformed = transform_targets(batch_y, target_means, target_vars)
    
            # Pass normalized inputs to the model
            outputs = model(batch_x3_norm_with_w, batch_x2_norm)
            loss = train_loss(outputs, batch_y_transformed)
            batch_mae = train_mae(outputs, batch_y_transformed)
            
            if i > 0:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                t2_1 = time.perf_counter()
                
                if i % 10 == 9:
                    print(f'batch {i+1}, time:{t2_1-t1_1:.3f}, loss: {loss:.4f}, mean_absolute_error: {batch_mae:.4f}')

        # Validation step
        model.eval()
        with torch.no_grad():
            for i, v_data in enumerate(valid_set):
                
                vx3d, vx2d, v_labels, v_w = v_data
                vx3d, vx2d, v_labels, v_w = vx3d.to(device), vx2d.to(device), v_labels.to(device), v_w.to(device)
                

                
                # Interpolate vertical wind and concatenate to x3d
                w_full = interpolate_w_to_full_levels(v_w)
                vx3d_with_w = torch.cat([vx3d, w_full], dim=-1)
                
                # Normalize validation inputs
                vx3d_norm_with_w, vx2d_norm, _ = normalizer.normalize(vx3d_with_w, vx2d)
                
                # Transform targets
                v_labels_transformed = transform_targets(v_labels, target_means, target_vars)
                
                # Pass normalized inputs to the model
                v_outputs = model(vx3d_norm_with_w, vx2d_norm)
                
                valid_loss.update(v_outputs, v_labels_transformed)
                valid_mae.update(v_outputs, v_labels_transformed)                

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

def test_model(model, test_set, normalizer, target_means, target_vars):
    logger.info('Test started...')    

    # Load the best model checkpoint
    best_chkpt = join(checkpoint_path, 'best_model.pth')
    assert isfile(best_chkpt), 'Checkpoint not found, testing failed!'
    checkpoint = torch.load(best_chkpt, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize loss and metric trackers
    test_loss = MeanSquaredError().to(device)
    test_mae = MeanAbsoluteError().to(device)

    y_true, y_pred = list(), list()
    
    t1 = time.perf_counter(), time.process_time()
    
    for i, data in enumerate(test_set):
        
        batch_x3, batch_x2, batch_y, batch_w = data
        batch_x3, batch_x2, batch_y, batch_w = batch_x3.to(device), batch_x2.to(device), batch_y.to(device), batch_w.to(device)
        
        
        # Interpolate vertical wind and concatenate to x3d
        w_full = interpolate_w_to_full_levels(batch_w)
        batch_x3_with_w = torch.cat([batch_x3, w_full], dim=-1)
        
        # Normalize test inputs
        batch_x3_norm_with_w, batch_x2_norm, _ = normalizer.normalize(batch_x3_with_w, batch_x2)
        
        # Transform targets
        batch_y_transformed = transform_targets(batch_y, target_means, target_vars)
        
        model.eval()
        with torch.no_grad():
            # Pass normalized inputs to the model
            outputs = model(batch_x3_norm_with_w, batch_x2_norm)
            
        # Calculate loss on transformed outputs
        loss = test_loss(outputs, batch_y_transformed)
        mae = test_mae(outputs, batch_y_transformed)
        
        # Inverse transform targets for saving
        outputs_original = inverse_transform_targets(outputs, target_means, target_vars)
        
        # Collect true and predicted values for further analysis
        y_true.append(batch_y.detach().cpu())
        y_pred.append(outputs_original.detach().cpu())

        if i % 100 == 99:
            print(f'batch {i+1} loss: {loss:.4f}, mean_absolute_error: {mae:.4f}')

    t2 = time.perf_counter(), time.process_time()
    
    # Concatenate all collected true and predicted values
    y_true = torch.cat(y_true, 0)
    y_pred = torch.cat(y_pred, 0)

    # Compute total test loss and mean absolute error
    total_test_loss = test_loss.compute()
    total_test_mae = test_mae.compute()

    print(f'Test time: {t2[0] - t1[0]:.2f} loss: {total_test_loss:.4f}, mean_absolute_error: {total_test_mae:.4f}')
    
    # Save true and predicted values to files for further analysis
    with open(join(test_path, 'y_true.pickle'), 'wb') as handle:
        pickle.dump(y_true, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(join(test_path, 'y_pred.pickle'), 'wb') as handle:
        pickle.dump(y_pred, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Also save target statistics for proper evaluation
    with open(join(test_path, 'target_stats.pickle'), 'wb') as handle:
        pickle.dump({'means': target_means.cpu(), 'vars': target_vars.cpu()}, handle, protocol=pickle.HIGHEST_PROTOCOL)



def get_column_data_with_disk_cache(input_filenames, output_filenames, shuffle=False, num_workers=0):
    logger.info(f"Creating dataset with triangle_id={args.triangle_id}, division_factor={args.triangle_division_factor}")

    icon_data = IconIterableDataset3DTendency(
        triangle_id=args.triangle_id,
        division_factor=args.triangle_division_factor,
        input_filenames=input_filenames,
        output_filenames=output_filenames,
        shuffle=shuffle,
        cache_dir='/tmp', 
        dtype='float32',
        total_cols=args.num_cells,      
    )

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
    
    # Find and sort input files
    INPUT_FILENAMES = glob.glob(join(args.dataset_input, '*_inputs_*.h5'))
    OUTPUT_FILENAMES = glob.glob(join(args.dataset_output, '*_tendencies_*.h5'))

    # Extract time indices from input and output file names
    input_time_indices = [float(re.search(r'_time_(\d+\.\d+)\.h5', f).group(1)) for f in INPUT_FILENAMES]
    output_time_indices = [float(re.search(r'_time_(\d+\.\d+)\.h5', f).group(1)) for f in OUTPUT_FILENAMES]

    # Sort the time indices
    sorted_input_time_indices = sorted(input_time_indices)
    sorted_output_time_indices = sorted(output_time_indices)

    # Ensure that sorted input and output time indices match
    assert sorted_input_time_indices == sorted_output_time_indices, "Input and output files have mismatched time indices"

    # Sort input and output files based on the sorted time indices
    sorted_input_files = [INPUT_FILENAMES[input_time_indices.index(time_index)] for time_index in sorted_input_time_indices]
    sorted_output_files = [OUTPUT_FILENAMES[output_time_indices.index(time_index)] for time_index in sorted_output_time_indices]
    
    # Set random seeds for reproducibility
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

    # Split data into train/val/test sets
    train_input_files = sorted_input_files[200:2000]
    train_output_files = sorted_output_files[200:2000]
    val_input_files = sorted_input_files[:160] + sorted_input_files[2020:2180]
    val_output_files = sorted_output_files[:160] + sorted_output_files[2020:2180]
    test_input_files = sorted_input_files[2220:]
    test_output_files = sorted_output_files[2220:]
    
    # Sample dataset if requested
    if args.percent < 1:
        train_indices = prng.choice(len(train_input_files), max(1, int(args.percent * len(train_input_files))), replace=False)
        train_input_files = [train_input_files[i] for i in train_indices]
        train_output_files = [train_output_files[i] for i in train_indices]
        
        val_indices = prng.choice(len(val_input_files), max(1, int(args.percent * len(val_input_files))), replace=False)
        val_input_files = [val_input_files[i] for i in val_indices]
        val_output_files = [val_output_files[i] for i in val_indices]
        
        test_indices = prng.choice(len(test_input_files), max(1, int(args.percent * len(test_input_files))), replace=False)
        test_input_files = [test_input_files[i] for i in test_indices]
        test_output_files = [test_output_files[i] for i in test_indices]

    # Load normalization stats
    stats_file = '/mydata/deepcloud/shared/h5_tendency_all/normalizer_stats_per_feat_updated.pickle'
    mean2d, var2d, mean3d, var3d = get_normalization_params(stats_file)
    
    # Create the normalizer
    normalizer = DataNormalizer(mean2d, var2d, mean3d, var3d, device=device)
    
    # Load target statistics
    target_stats_file = '/mydata/deepcloud/shared/h5_tendency_all/normalizer_stats_per_feat_y2_no_temp.pickle'
    with open(target_stats_file, 'rb') as f:
        target_stats = pickle.load(f)

    target_means = torch.tensor(target_stats['mean'], dtype=torch.float32).to(device)
    target_vars = torch.tensor(target_stats['var'], dtype=torch.float32).to(device)
    
    # Initialize model
    model = get_model()
    num_params = count_parameters(model)
    print(f"The model has {num_params:,} trainable parameters.")
    
    # Restore RNG states after model initialization
    torch.set_rng_state(torch_rng_state)
    np.random.set_state(np_rng_state)
    random.setstate(random_rng_state)

    if args.train:
        tr1 = time.perf_counter(), time.process_time()                        

        train_loader = get_column_data_with_disk_cache(train_input_files, train_output_files, shuffle=True, num_workers=args.num_workers)
        val_loader = get_column_data_with_disk_cache(val_input_files, val_output_files, shuffle=False, num_workers=args.num_workers)
        
        
        # Train the model
        train_model(model, train_loader, val_loader, normalizer, target_means, target_vars)

        tr2 = time.perf_counter(), time.process_time()
        print(f'Training time: Real time: {tr2[0] - tr1[0]:.2f}, CPU time: {tr2[1]-tr1[1]}')
              
    
    if args.test:
        test_loader = get_column_data_with_disk_cache(test_input_files, test_output_files, shuffle=False, num_workers=args.num_workers)
        test_model(model, test_loader, normalizer, target_means, target_vars)
        
        print("\nTesting completed. Run the evaluation script to see detailed metrics.")
    
    logger.info('Code ended!')

if __name__ == '__main__':
    main() 