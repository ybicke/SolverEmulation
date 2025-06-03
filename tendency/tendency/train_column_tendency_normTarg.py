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

from data_loaders_tendency import IconColumnIterableDataset
from data_utils import DataNormalizer


from memory_efficient_baseline_function import precompute_train_target_mean




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


parser = argparse.ArgumentParser(description='Train Transformer models.')
parser.add_argument('--model', type=str, default='vit', help='Name of the model to be trained')
parser.add_argument('--dataset_input', type=str, help='Path to the input dataset')
parser.add_argument('--dataset_output', type=str, help='Path to the output dataset')
parser.add_argument('--save', type=str, required=True, help='Path to save the result')
parser.add_argument('--percent', type=float, default=None, help='Percentage of data to use')
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
parser.add_argument('--hidden-dim', type=int, default=256, help='hidden dimension')
parser.add_argument('--layers', type=int, default=4, help='layers')
parser.add_argument('--heads', type=int, default=6, help='heads')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
parser.add_argument('--afno-sparsity-threshold', type=float, default=0.01, help='Sparsity threshold for AFNO')
parser.add_argument('--hard-thresholding-fraction', type=float, default=1, help='hard thresholding fraction AFNO')
parser.add_argument('--cutoff-frequency', type=float, default=0.1, help='cutoff frequency low pass filtering in AFNO')
parser.add_argument('--zero-freq-indices', nargs='+', type=int, default=None, help='Zero frequency indices to zero out')



# Add GNN specific arguments
parser.add_argument('--max-skip', type=int, default=3, help='Maximum skip distance for hierarchical edges in GNN')
parser.add_argument('--fully-connected', action=argparse.BooleanOptionalAction, default=False, help='Use fully connected graph for GNN')
parser.add_argument('--edge-channels-in', type=int, default=1, help='Number of edge feature channels for GNN')
parser.add_argument('--emb-dropout', type=float, default=0.0, help='Dropout rate for embeddings in GNN')
parser.add_argument('--channel-3d', type=int, default=13, help='Number of 3D input channels for GNN')
parser.add_argument('--channel-2d', type=int, default=3, help='Number of 2D input channels for GNN')
parser.add_argument('--channels-out', type=int, default=7, help='Number of output channels for GNN')

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
    

def get_model(model_name, is_test):
    logger.info('Preparing the model...')
    
    if model_name == 'afno_tendency':
        from afno_column_clean_tendency import AFNONet
        
        model = AFNONet(
            num_cells=args.num_cells,
            patch_size=args.patch_size,
            embed_dim=args.hidden_dim,
            # mlp_dim=args.vit_hidden_dim,
            depth=args.layers, #num blocks
            dropout=args.dropout, # used in the mlp
            device=device,
            is_test=args.test,  
            sparsity_threshold=args.afno_sparsity_threshold,  
            hard_thresholding_fraction = args.hard_thresholding_fraction,
        ).to(device)
    
    elif model_name == 'gnn_tendency':
        from gnn import AtmosphericColumnGNN
        
        model = AtmosphericColumnGNN(
            embed_dim=args.hidden_dim,
            depth=args.layers,
            dropout=args.dropout,
            max_skip=args.max_skip,
            emb_dropout=args.emb_dropout,
            channel_3d=args.channel_3d,  
            channel_2d=args.channel_2d,
            channels_out=args.channels_out, 
            edge_channels_in=args.edge_channels_in,
            fully_connected=args.fully_connected,
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



def interpolate_w_to_full_levels_tensor(w):
    """Linear interpolation"""
    batch_size, num_half_levels, _ = w.shape
    num_full_levels = num_half_levels -1
    
    w_full = torch.zeros((batch_size, num_full_levels, 1), device=w.device)
    w_full[:, :, :] = 0.5 * (w[:, :-1, :] + w[:, 1:, :])
       
    return w_full




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
        logger.info(f'Training will continue from epoch: {init_epoch}/{args.num_epoch}')
    else:
        init_epoch = 0

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
            
            batch_x3, batch_x2, batch_y, batch_w = data
            batch_x3, batch_x2, batch_y, batch_w = batch_x3.to(device), batch_x2.to(device), batch_y.to(device), batch_w.to(device)
            

            # Interpolate w from half levels to full levels here we add the w to the 3d finally
            w_full = interpolate_w_to_full_levels_tensor(batch_w)
            batch_x3 = torch.cat([batch_x3, w_full], dim=-1)
            
            # Normalize the inputs using the DataNormalizer
            batch_x3_norm, batch_x2_norm, batch_x2_orig = normalizer.normalize(batch_x3, batch_x2)
            
            # Transform targets
            batch_y_transformed = transform_targets(batch_y, target_means, target_vars)
    
            # Pass normalized inputs to the model
            outputs = model(batch_x3_norm, batch_x2_norm)
            loss = train_loss(outputs, batch_y_transformed)
            batch_mae = train_mae(outputs, batch_y_transformed)
            
            
            if i > 0:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                t2_1 = time.perf_counter()
                
                if i % 100 == 99:
                    print(f'batch {i+1}, time:{t2_1-t1_1:.3f}, loss: {loss:.4f}, mean_absolute_error: {batch_mae:.4f}')
                

        # Validation step
        model.eval()
        
        with torch.no_grad():
            for i, v_data in enumerate(valid_set):
                
                vx3d, vx2d, v_labels, v_w = v_data
                vx3d, vx2d, v_labels, v_w = vx3d.to(device), vx2d.to(device), v_labels.to(device), v_w.to(device)
                
                # Interpolate w from half levels to full levels
                w_full = interpolate_w_to_full_levels_tensor(v_w)
                vx3d = torch.cat([vx3d, w_full], dim=-1)
                
                # Normalize validation inputs
                vx3d_norm, vx2d_norm, vx2d_orig = normalizer.normalize(vx3d, vx2d)
                
                # Transform targets
                v_labels_transformed = transform_targets(v_labels, target_means, target_vars)
                
                # Pass normalized inputs to the model
                v_outputs = model(vx3d_norm, vx2d_norm)        
                
                # two different losses         
                valid_loss.update(v_outputs, v_labels_transformed)
                valid_mae.update(v_outputs, v_labels_transformed)                

        # Compute total losses and metrics - calculated once per epoch
        total_train_loss = train_loss.compute()
        total_valid_loss = valid_loss.compute()
        total_train_mae = train_mae.compute()
        total_valid_mae = valid_mae.compute()

        t2 = time.perf_counter()

        # Log basic metrics to W&B (single call per epoch)
        wandb.log({
            'epoch': epoch_number,
            'loss': total_train_loss.item(),
            'val_loss': total_valid_loss.item(),
            'mean_absolute_error': total_train_mae.item(),
            'val_mean_absolute_error': total_valid_mae.item()
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
    
    # Initialize loss and metric trackers for basic monitoring only
    test_loss = MeanSquaredError().to(device)
    test_mae = MeanAbsoluteError().to(device)

    y_true, y_pred = list(), list()
    
    t1 = time.perf_counter(), time.process_time()
    
    for i, data in enumerate(test_set):
        
        batch_x3, batch_x2, batch_y, batch_w = data
        batch_x3, batch_x2, batch_y, batch_w = batch_x3.to(device), batch_x2.to(device), batch_y.to(device), batch_w.to(device)
        
        # Interpolate w from half levels to full levels
        w_full = interpolate_w_to_full_levels_tensor(batch_w)
        batch_x3 = torch.cat([batch_x3, w_full], dim=-1)
        
        # Normalize test inputs
        batch_x3_norm, batch_x2_norm, batch_x2_orig = normalizer.normalize(batch_x3, batch_x2)
        
        # Transform targets
        batch_y_transformed = transform_targets(batch_y, target_means, target_vars)
        
        model.eval()
        with torch.no_grad():
            # Pass normalized inputs to the model
            outputs = model(batch_x3_norm, batch_x2_norm)
            
        # Inverse transform targets
        outputs_original = inverse_transform_targets(outputs, target_means, target_vars)
        
        # Collect true and predicted values for further analysis
        y_true.append(batch_y.detach().cpu())
        y_pred.append(outputs_original.detach().cpu())

        # Calculate and log basic metrics for monitoring progress
        loss = test_loss(outputs, batch_y_transformed)
        mae = test_mae(outputs, batch_y_transformed)
        
        if i % 1000 == 999:
            print(f'batch {i+1} loss: {loss:.4f}, mean_absolute_error: {mae:.4f}')

    t2 = time.perf_counter(), time.process_time()
    
    # Concatenate all collected true and predicted values
    y_true = torch.cat(y_true, 0)
    y_pred = torch.cat(y_pred, 0)

    # Compute total test loss and mean absolute error
    total_test_loss = test_loss.compute()
    total_test_mae = test_mae.compute()

    print(f'Test time: {t2[0] - t1[0]:.2f} loss: {total_test_loss:.4f}, mean_absolute_error: {total_test_mae:.4f}')
    
    # Save true and predicted values to files for further analysis and visualization
    with open(join(test_path, 'y_true.pickle'), 'wb') as handle:
        pickle.dump(y_true, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(join(test_path, 'y_pred.pickle'), 'wb') as handle:
        pickle.dump(y_pred, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Also save target statistics for proper evaluation
    with open(join(test_path, 'target_stats.pickle'), 'wb') as handle:
        pickle.dump({'means': target_means.cpu(), 'vars': target_vars.cpu()}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    logger.info(f'Results saved to {test_path}. Run evaluation script for detailed metrics.')


def test_loading_time(input_filenames, output_filenames, iter=10):
    logger.info('Test Loading time started...')
    dataset = get_column_data_with_disk_cache(input_filenames, output_filenames, shuffle=True)
        
    for it in range(iter):
        t1 = time.perf_counter(), time.process_time()
        for i, data in enumerate(dataset):
            batch_x3, batch_x2, batch_y = data
            batch_x3, batch_x2, batch_y = batch_x3.to(device), batch_x2.to(device), batch_y.to(device)
        t2 = time.perf_counter(), time.process_time()
        logger.info(f'Iteration: {it}: Real time: {t2[0] - t1[0]:.2f}, CPU time: {t2[1]-t1[1]}')
        import gc
        gc.collect()    
        
        
def get_column_data_with_disk_cache(input_filenames, output_filenames, subsample=args.subsample, shuffle=False, num_workers=0):
    icon_data = IconColumnIterableDataset(input_filenames, output_filenames, subsample=subsample, cache_dir='/tmp', shuffle=shuffle)

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


def transform_targets(batch_y, means, variances, k=4, min_scale=1e-6):
    """Standardize targets to zero-mean, unit-variance representation.
    
    Args:
        batch_y: Raw targets (B,L,C) in physical units
        means, variances: Per-channel statistics (C,)
        k: Scale factor for normalizing to approx. [-1,1] range
        min_scale: Lower bound for scaling to avoid div-by-zero
    """
    scale = torch.clamp(k * torch.sqrt(variances), min=min_scale)
    means = means.view(1, 1, -1).expand_as(batch_y)
    scale = scale.view(1, 1, -1).expand_as(batch_y)
    
    return (batch_y - means) / scale


def inverse_transform_targets(y_norm, means, variances, k=4, min_scale=1e-6):
    """Convert normalized values back to physical units."""
    scale = torch.clamp(k * torch.sqrt(variances), min=min_scale)
    means = means.view(1, 1, -1).expand_as(y_norm)
    scale = scale.view(1, 1, -1).expand_as(y_norm)
    
    return y_norm * scale + means


def main():
        
    logger.info('Code started...')

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
    
    train_input_files = sorted_input_files[200:2000]
    train_output_files = sorted_output_files[200:2000]
    val_input_files = sorted_input_files[:160] + sorted_input_files[2020:2180]
    val_output_files = sorted_output_files[:160] + sorted_output_files[2020:2180]
    test_input_files = sorted_input_files[2220:]
    test_output_files = sorted_output_files[2220:]
    
    # Sample dataset
    if args.percent is not None and args.percent < 1:
        train_indices = prng.choice(len(train_input_files), max(1, int(args.percent * len(train_input_files))), replace=False)
        train_input_files = [train_input_files[i] for i in train_indices]
        train_output_files = [train_output_files[i] for i in train_indices]
        
        val_indices = prng.choice(len(val_input_files), max(1, int(args.percent * len(val_input_files))), replace=False)
        val_input_files = [val_input_files[i] for i in val_indices]
        val_output_files = [val_output_files[i] for i in val_indices]
        
        test_indices = prng.choice(len(test_input_files), max(1, int(args.percent * len(test_input_files))), replace=False)
        test_input_files = [test_input_files[i] for i in test_indices]
        test_output_files = [test_output_files[i] for i in test_indices]


    # stats_file = join(args.dataset_input, 'normalizer_stats_per_feat_updated.pickle')
    stats_file = '/mydata/deepcloud/shared/h5_tendency_all/normalizer_stats_per_feat_updated.pickle'
    mean2d, var2d, mean3d, var3d = get_normalization_params(stats_file)
    
    # Create the normalizer
    normalizer = DataNormalizer(mean2d, var2d, mean3d, var3d, device=device)
    
    # Load target statistics!!!
    # target_stats_file = join(args.dataset_input, 'normalizer_stats_per_feat_y2_no_temp.pickle')
    target_stats_file = '/mydata/deepcloud/shared/h5_tendency_all/normalizer_stats_per_feat_y2_no_temp.pickle'
    with open(target_stats_file, 'rb') as f:
        target_stats = pickle.load(f)

    target_means = torch.tensor(target_stats['mean'], dtype=torch.float32).to(device)  # Shape: (num_features,)
    target_vars = torch.tensor(target_stats['var'], dtype=torch.float32).to(device)    # Shape: (num_features,)
    
    
    
    model = get_model(args.model, args.test)
    num_params = count_parameters(model)
    print(f"The model has {num_params:,} trainable parameters.")
    
    # Restore RNG states after model initialization
    torch.set_rng_state(torch_rng_state)
    np.random.set_state(np_rng_state)
    random.setstate(random_rng_state)



    if args.train:
        tr1 = time.perf_counter(), time.process_time()                        

        train_loader = get_column_data_with_disk_cache(train_input_files, train_output_files, shuffle=True)
        val_loader = get_column_data_with_disk_cache(val_input_files, val_output_files, shuffle=False, subsample=1.0)
        
        
        # Compute train target mean and save it to test path for later use during testing
        train_target_mean_file = join(test_path, 'train_target_mean_all_train_data.pickle')
        print('Computing train_target_mean from training data...')
        train_target_mean = precompute_train_target_mean(train_loader)
        
        # Save train_target_mean for later use during testing
        with open(train_target_mean_file, 'wb') as handle:
            pickle.dump(train_target_mean, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Saved train_target_mean to {train_target_mean_file}')
        
   
        train_model(model, train_loader, val_loader, normalizer, target_means, target_vars)

        tr2 = time.perf_counter(), time.process_time()
        print(f'Training time: Real time: {tr2[0] - tr1[0]:.2f}, CPU time: {tr2[1]-tr1[1]}')
              
    
    if args.test:
        test_loader = get_column_data_with_disk_cache(test_input_files, test_output_files, shuffle=False)
        test_model(model, test_loader, normalizer, target_means, target_vars)
        
        print("\nTesting completed. Run the evaluation script to see detailed metrics:")
        print(f"python visualize_results/visualize_tendency_mae_baseline_vs_model.py --model_path {test_path}")
    
    logger.info('Code ended!')

if __name__ == '__main__':
    main()
    
