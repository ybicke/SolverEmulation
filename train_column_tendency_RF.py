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

import numpy as np
import matplotlib.pyplot as plt

from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from torchinfo import summary
import torch.autograd.profiler as profiler

# ----- NEW: scikit-learn imports -----
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# If you want W&B, you can keep them for logging your RF metrics
import wandb

from data_loaders_tendency import IconColumnIterableDataset


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

parser = argparse.ArgumentParser(description='Train Random Forest on ICON data.')
parser.add_argument('--model', type=str, default='rf', help='Name of the model to be trained (we only do rf here).')
parser.add_argument('--dataset_input', type=str, help='Path to the input dataset')
parser.add_argument('--dataset_output', type=str, help='Path to the output dataset')
parser.add_argument('--save', type=str, required=True, help='Path to save the result')
parser.add_argument('--percent', type=float, default=1.0, help='Percent of data to train on')
parser.add_argument('--subsample', type=float, default=None, help='Subsampling rate')
parser.add_argument('--num-workers', type=int, default=os.cpu_count(), help='Number of workers to load data')
parser.add_argument('--prefetch-factor', type=int, default=2, help='Prefetch factor')
parser.add_argument('--wandb-mode', type=str, default='disabled',
                    choices={'online', 'offline', 'disabled'}, help='W&B mode')
parser.add_argument('--train', action=argparse.BooleanOptionalAction, default=True, help='Specify if training takes place')
parser.add_argument('--test', action=argparse.BooleanOptionalAction, default=True, help='Specify if test takes place')
parser.add_argument('--shuffle', action=argparse.BooleanOptionalAction, default=True, help='Shuffling the train dataset')
parser.add_argument('--batch-size', type=int, default=4, help='Batch size for data loading (not used in RF training)')
# parser.add_argument('--min-samples-leaf', type=int, default=2, help='Minimum number of samples required to be at a leaf node')
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
        


# -----------------------
# Load Input Statistics
# -----------------------
def get_input_normalization_params(stats_file):
    """
    Loads mean2d, var2d, mean3d, var3d from a pickle,
    returns them as CPU Tensors for convenience.
    """
    with open(stats_file, 'rb') as f:
        stats = pickle.load(f)
    mean2d = torch.tensor(stats['mean2d'], dtype=torch.float32)
    var2d = torch.tensor(stats['var2d'], dtype=torch.float32)
    mean3d = torch.tensor(stats['mean3d'], dtype=torch.float32)
    var3d = torch.tensor(stats['var3d'], dtype=torch.float32)
    return mean2d, var2d, mean3d, var3d


# -----------------------
# Load Target Statistics
# -----------------------
def get_output_normalization_params(target_stats_file):
    """
    Suppose your output stats file has 'mean' and 'var' keys in shape [n_outputs].
    """
    with open(target_stats_file, 'rb') as f:
        target_stats = pickle.load(f)
    target_mean = torch.tensor(target_stats['mean'], dtype=torch.float32)
    target_var = torch.tensor(target_stats['var'], dtype=torch.float32)
    return target_mean, target_var


# -------------------------------------------------------------------------
# Random Forest "train" function
# -------------------------------------------------------------------------
def train_random_forest(
    train_input_files, train_output_files,
    val_input_files, val_output_files,
    mean2d, var2d, mean3d, var3d,
    target_mean, target_var
):
    


    """
    Collect train data, normalize, fit RF, validate.
    """
    wandb.init(
        project='deepcloud-yves',
        name=save_id,
        id=save_id,
        config=wandb_config,
        resume='allow',
        mode=args.wandb_mode
    )

    # 1) Training Data
    train_loader = get_column_data_with_disk_cache(
        train_input_files, train_output_files,
        shuffle=True, num_workers=args.num_workers
    )
    
    start_time = time.time()
    X_train, y_train = collect_dataset_from_dataloader(
        dataloader=train_loader,
        mean2d=mean2d, var2d=var2d,
        mean3d=mean3d, var3d=var3d,
        target_mean=target_mean, target_var=target_var
    )
    end_time = time.time()
    logger.info(f"Data loading completed in {end_time - start_time:.2f} seconds.")

    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)

    logger.info("Initializing RandomForestRegressor...")

        
    # min_samples_leaf must be 10^-2% (i.e., 0.001) times 'n'.
    min_samples_for_leaf = int(0.01 * len(X_train))

    rf = RandomForestRegressor(
        n_estimators=10,
        min_samples_leaf=min_samples_for_leaf,
        max_depth=None,     
        random_state=42,
        verbose=2,
        n_jobs=-1
    )
    
    #rf = RandomForestRegressor(
    #    n_estimators=args.n_estimators,
    #    max_depth=args.max_depth,
    #    # min_samples_leaf=args.min_samples_leaf,
    #    random_state=42,
    #    verbose=2,  # Add verbosity for monitoring
    #    n_jobs=-1
    #)

    logger.info(f"Fitting RandomForestRegressor on shape X={X_train.shape}, y={y_train.shape}")
    t0 = time.time()
    rf.fit(X_train, y_train)
    t1 = time.time()
    logger.info(f"RF fit done in {t1 - t0:.2f} seconds.")

    # 2) Validation Data
    val_loader = get_column_data_with_disk_cache(
        val_input_files, val_output_files,
        shuffle=False, num_workers=args.num_workers
    )
    X_val, y_val = collect_dataset_from_dataloader(
        dataloader=val_loader,
        mean2d=mean2d, var2d=var2d,
        mean3d=mean3d, var3d=var3d,
        target_mean=target_mean, target_var=target_var
    )

    y_pred_val = rf.predict(X_val)
    mse_val = mean_squared_error(y_val, y_pred_val)
    mae_val = mean_absolute_error(y_val, y_pred_val)
    logger.info(f"Validation MSE: {mse_val:.4f}, MAE: {mae_val:.4f}")
    wandb.log({"val_mse": mse_val, "val_mae": mae_val})

    # Save RF
    rf_path = join(checkpoint_path, 'rf_model.pkl')
    with open(rf_path, 'wb') as f:
        pickle.dump(rf, f)
    logger.info(f"Saved RF model to {rf_path}")
    return rf



# -------------------------------------------------------------------------
# Random Forest "test" function
# -------------------------------------------------------------------------
def test_random_forest(
    rf,
    test_input_files, test_output_files,
    mean2d, var2d, mean3d, var3d,
    target_mean, target_var
):
    logger.info("Starting Random Forest test...")

    test_loader = get_column_data_with_disk_cache(
        test_input_files, test_output_files,
        shuffle=False, num_workers=args.num_workers
    )
    X_test, y_test = collect_dataset_from_dataloader(
        dataloader=test_loader,
        mean2d=mean2d, var2d=var2d,
        mean3d=mean3d, var3d=var3d,
        target_mean=target_mean, target_var=target_var
    )

    y_pred_test = rf.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    logger.info(f"Test MSE: {mse_test:.4f}, MAE: {mae_test:.4f}")

    # Save predictions for further analysis
    with open(join(test_path, 'y_true_test.pkl'), 'wb') as f:
        pickle.dump(y_test, f)
    with open(join(test_path, 'y_pred_test.pkl'), 'wb') as f:
        pickle.dump(y_pred_test, f)


# -------------------------------------------------------------------------
# A. Data Loader: Reuse your existing get_column_data_with_disk_cache
# -------------------------------------------------------------------------
        
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


# -------------------------------------------------------------------------
# Collect / Normalize Data
# -------------------------------------------------------------------------
def collect_dataset_from_dataloader(
    dataloader,
    mean2d, var2d,
    mean3d, var3d,
    target_mean, target_var
):
    """
    Loads ALL data from the DataLoader into memory (X, Y).
    Applies normalization to (x3d, x2d, y) before flattening.
    Returns X (np.array), Y (np.array).

    mean2d: shape [n_2D_features]
    var2d:  shape [n_2D_features]
    mean3d: shape [n_3D_features]
    var3d:  shape [n_3D_features]
    target_mean: shape [n_output_features]
    target_var:  shape [n_output_features]
    """
    X_list = []
    Y_list = []

    for batch in dataloader:
        if len(batch) == 4:
            batch_x3, batch_x2, batch_y, batch_w = batch
        else:
            raise ValueError("Dataset must yield 4 items: x3d, x2d, y, w")

        # Move to CPU
        batch_x3 = batch_x3.cpu()
        batch_x2 = batch_x2.cpu()
        batch_y  = batch_y.cpu()
        batch_w  = batch_w.cpu()

        # Optional interpolation for w
        w_full = interpolate_w_to_full_levels_tensor(batch_w)
        # Now batch_x3 shape: [B, 70, ???]
        batch_x3 = torch.cat([batch_x3, w_full], dim=-1)  # e.g. => [B,70,(13+1)] = [B,70,14]

        # -------------------------------------------------------
        # 1) Normalize x3d: shape [B,70, n_3D_features]
        #    Check that last dim == mean3d.shape[0]
        # -------------------------------------------------------
        # (x - mean3d)/sqrt(var3d) => broadcast on last dim
        batch_x3 = (batch_x3 - mean3d) / torch.sqrt(var3d)

        # -------------------------------------------------------
        # 2) Normalize x2d: shape [B, n_2D_features]
        # -------------------------------------------------------
        batch_x2 = (batch_x2 - mean2d) / torch.sqrt(var2d)

        # -------------------------------------------------------
        # 3) Normalize y: shape [B,70, n_output_features]
        # -------------------------------------------------------
        batch_y = (batch_y - target_mean) / torch.sqrt(target_var)

        # Flatten input:
        # E.g. replicate x2d across the 70 levels if that's what you do:
        x2d_to_70 = batch_x2.unsqueeze(1).repeat(1, batch_x3.shape[1], 1)  # [B,70, n_2D_features]
        x3d_merged = torch.cat((batch_x3, x2d_to_70), dim=-1)             # [B,70, (14 + n_2D_features)]

        # Flatten each sample -> shape [B, 70*(14 + n_2D_features)]
        X_flat = x3d_merged.view(batch_x3.size(0), -1).numpy()

        # Flatten target -> shape [B, 70*n_output_features]
        Y_flat = batch_y.view(batch_y.size(0), -1).numpy()

        X_list.append(X_flat)
        Y_list.append(Y_flat)

    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    return X, Y



# -------------------------------------------------------------------------
# main()
# -------------------------------------------------------------------------
def main():
    logger.info("Code started...")

    # 1) Gather .h5 input & output files
    INPUT_FILENAMES = glob.glob(join(args.dataset_input, '*_inputs_*.h5'))
    OUTPUT_FILENAMES = glob.glob(join(args.dataset_output, '*_tendencies_*.h5'))

    input_time_indices = [float(re.search(r'_time_(\d+\.\d+)\.h5', f).group(1)) for f in INPUT_FILENAMES]
    output_time_indices = [float(re.search(r'_time_(\d+\.\d+)\.h5', f).group(1)) for f in OUTPUT_FILENAMES]

    # Sort by time
    sorted_input_time_indices = sorted(input_time_indices)
    sorted_output_time_indices = sorted(output_time_indices)
    assert sorted_input_time_indices == sorted_output_time_indices, "Mismatch in time indices!"

    sorted_input_files = [INPUT_FILENAMES[input_time_indices.index(t)] for t in sorted_input_time_indices]
    sorted_output_files = [OUTPUT_FILENAMES[output_time_indices.index(t)] for t in sorted_output_time_indices]

    # 2) Split train / val / test
    train_input_files = sorted_input_files[200:2000]
    train_output_files = sorted_output_files[200:2000]
    val_input_files = sorted_input_files[:160] + sorted_input_files[2020:2180]
    val_output_files = sorted_output_files[:160] + sorted_output_files[2020:2180]
    test_input_files = sorted_input_files[2220:]
    test_output_files = sorted_output_files[2220:]

    # 3) Subsample if needed
    if args.percent < 1.0:
        train_indices = prng.choice(len(train_input_files), max(1, int(args.percent * len(train_input_files))), replace=False)
        train_input_files = [train_input_files[i] for i in train_indices]
        train_output_files = [train_output_files[i] for i in train_indices]

        val_indices = prng.choice(len(val_input_files), max(1, int(args.percent * len(val_input_files))), replace=False)
        val_input_files = [val_input_files[i] for i in val_indices]
        val_output_files = [val_output_files[i] for i in val_indices]

        test_indices = prng.choice(len(test_input_files), max(1, int(args.percent * len(test_input_files))), replace=False)
        test_input_files = [test_input_files[i] for i in test_indices]
        test_output_files = [test_output_files[i] for i in test_indices]

    # 4) Load Input & Output Normalization Stats
    #   Adjust the paths to your actual pickle files for input + target stats
    stats_file = '/mydata/deepcloud/shared/h5_tendency_all/normalizer_stats_per_feat_updated1.pickle'
    mean2d, var2d, mean3d, var3d = get_input_normalization_params(stats_file)

    target_stats_file = '/mydata/deepcloud/shared/h5_tendency_all/normalizer_stats_per_feat_y2_no_temp.pickle'

    target_mean, target_var = get_output_normalization_params(target_stats_file)

    # 5) Train or Load
    rf_model = None
    rf_path = join(checkpoint_path, 'rf_model.pkl')

    if args.train:
        rf_model = train_random_forest(
            train_input_files, train_output_files,
            val_input_files, val_output_files,
            mean2d, var2d, mean3d, var3d,
            target_mean, target_var
        )
    else:
        if isfile(rf_path):
            with open(rf_path, 'rb') as f:
                rf_model = pickle.load(f)
            logger.info(f"Loaded RF model from {rf_path}")
        else:
            raise FileNotFoundError("No existing RF model found to load.")

    # 6) Test
    if args.test and rf_model is not None:
        test_random_forest(
            rf_model,
            test_input_files, test_output_files,
            mean2d, var2d, mean3d, var3d,
            target_mean, target_var
        )

    logger.info("Code ended!")


if __name__ == '__main__':
    main()
