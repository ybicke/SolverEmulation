import h5py
import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import glob
import os
from tqdm import tqdm
import pickle
from datetime import datetime
import time
import psutil
import argparse
import math
import torch.multiprocessing as mp


def parse_args():
    parser = argparse.ArgumentParser(description='Calculate dataset statistics')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Directory containing the input h5 files')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save the statistics file')
    parser.add_argument('--batch_size', type=int, default=10,
                      help='Number of files to process in each batch')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of worker processes for data loading')
    parser.add_argument('--pickle_name', type=str, default='normalizer_stats_per_feat.pickle',
                      help='Name of the pickle output file')
    parser.add_argument('--txt_name', type=str, default='normalizer_stats_summary.txt',
                      help='Name of the text output file')
    return parser.parse_args()


class WeatherDataset(IterableDataset):
    def __init__(self, filenames):
        super(WeatherDataset).__init__()
        self.filenames = filenames

    def read_file(self, filename):
        try:
            with h5py.File(filename, 'r') as h:
                w = torch.tensor(h['w'][:], dtype=torch.float32)
                x2d = torch.tensor(h['x2d'][:], dtype=torch.float32)
                x3d = torch.tensor(h['x3d'][:], dtype=torch.float32)
            return w, x2d, x3d
        except Exception as e:
            print(f"Error reading file {filename}: {e}")
            return None, None, None

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process data loading
            iter_start = 0
            iter_end = len(self.filenames)
        else:
            # In a worker process
            per_worker = int(math.ceil(len(self.filenames) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.filenames))

        for filename in self.filenames[iter_start:iter_end]:
            w, x2d, x3d = self.read_file(filename)
            if w is None:
                continue  # Skip if the file could not be read
            yield w, x2d, x3d



def calculate_statistics(loader, device, total_files):
    start_time = time.time()
    
    # Initialize sums for each feature
    w_sum = torch.zeros(1, dtype=torch.float32, device=device)      # 1 feature
    w_squared_sum = torch.zeros(1, dtype=torch.float32, device=device)
    
    x2d_sum = torch.zeros(3, dtype=torch.float32, device=device)    # 3 features
    x2d_squared_sum = torch.zeros(3, dtype=torch.float32, device=device)
    
    x3d_sum = torch.zeros(8, dtype=torch.float32, device=device)    # 8 features
    x3d_squared_sum = torch.zeros(8, dtype=torch.float32, device=device)
    
    total_w_samples = 0
    total_2d_samples = 0
    total_3d_samples = 0
    timesteps_processed = 0
    timesteps_per_update = 200  # Print every 200 timesteps

    for batch_idx, (w, x2d, x3d) in enumerate(loader):
        # Move data to device
        w = w.to(device)      # shape: (1, 81920, 71, 1)
        x2d = x2d.to(device)  # shape: (1, 81920, 3)
        x3d = x3d.to(device)  # shape: (1, 81920, 70, 8)
        
        current_batch_timesteps = w.size(0)  # Number of timesteps in this batch
        timesteps_processed += current_batch_timesteps

         # Flatten all dimensions except features
        w_flat = w.reshape(-1, 1)          # Combine batch, cells, and height dimensions
        x2d_flat = x2d.reshape(-1, 3)      # Combine batch and cells dimensions
        x3d_flat = x3d.reshape(-1, 8)      # Combine batch, cells, and height dimensions
        
        # Update statistics
        w_sum += torch.sum(w_flat, dim=0)
        w_squared_sum += torch.sum(w_flat ** 2, dim=0)
        total_w_samples += w_flat.size(0)

        x2d_sum += torch.sum(x2d_flat, dim=0)
        x2d_squared_sum += torch.sum(x2d_flat ** 2, dim=0)
        total_2d_samples += x2d_flat.size(0)

        x3d_sum += torch.sum(x3d_flat, dim=0)
        x3d_squared_sum += torch.sum(x3d_flat ** 2, dim=0)
        total_3d_samples += x3d_flat.size(0)

        # Print progress every 200 timesteps
        if timesteps_processed // timesteps_per_update > (timesteps_processed - current_batch_timesteps) // timesteps_per_update:
            print(f"\nProgress update at {datetime.now()}:", flush=True)
            progress = (timesteps_processed / total_files) * 100
            print(f"Processed: {timesteps_processed}/{total_files} timesteps ({progress:.2f}%)", flush=True)

    # Calculate final statistics
    mean_w = w_sum / total_w_samples
    var_w = (w_squared_sum / total_w_samples) - (mean_w ** 2)
    
    mean_2d = x2d_sum / total_2d_samples
    var_2d = (x2d_squared_sum / total_2d_samples) - (mean_2d ** 2)
    
    mean_3d = x3d_sum / total_3d_samples
    var_3d = (x3d_squared_sum / total_3d_samples) - (mean_3d ** 2)

    return {
        'mean_w': mean_w.cpu().numpy(),    # shape: (1,)
        'var_w': var_w.cpu().numpy(),      # shape: (1,)
        'mean2d': mean_2d.cpu().numpy(),   # shape: (3,)
        'var2d': var_2d.cpu().numpy(),     # shape: (3,)
        'mean3d': mean_3d.cpu().numpy(),   # shape: (8,)
        'var3d': var_3d.cpu().numpy()      # shape: (8,)
    }

def save_statistics(stats, output_dir, start_time, pickle_name, txt_name):
    # Save raw statistics
    output_file = os.path.join(output_dir, pickle_name)
    with open(output_file, 'wb') as f:
        pickle.dump(stats, f)

    # Prepare human-readable output
    output_stats = []
    
    # Header
    header = f"\nFeature-wise statistics summary"
    timestamp = f"Generated on: {datetime.now()}"
    output_stats.extend([header, timestamp, ""])
    
    # w statistics
    w_stats = "Target Variable:"
    w_details = f"w (Vertical velocity) [m/s]:\n  Mean: {stats['mean_w'][0]:.6f}\n  Variance: {stats['var_w'][0]:.6f}"
    output_stats.extend([w_stats, w_details, ""])
    
    # x2d statistics
    x2d_stats = "Surface-level Features (x2d):"
    output_stats.append(x2d_stats)
    x2d_features = ['pres_sfc - Surface pressure [Pa]', 
                    'cosmu0 - Cosine of solar zenith angle [-]', 
                    'qv_s - Surface water vapor specific humidity [kg/kg]']
    
    for i, feature_name in enumerate(x2d_features):
        feature_stats = f"{feature_name}:\n  Mean: {stats['mean2d'][i]:.6f}\n  Variance: {stats['var2d'][i]:.6f}"
        output_stats.append(feature_stats)
    output_stats.append("")
    
    # x3d statistics
    x3d_stats = "3D Features (x3d):"
    output_stats.append(x3d_stats)
    x3d_features = ['u - Zonal wind [m/s]',
                    'v - Meridional wind [m/s]',
                    'pres - Pressure [Pa]',
                    'geopot - Geopotential [m²/s²]',
                    'qc - Cloud water content [kg/kg]',
                    'qi - Cloud ice content [kg/kg]',
                    'qv - Water vapor specific humidity [kg/kg]',
                    'clc - Cloud cover fraction [-]']
    
    for i, feature_name in enumerate(x3d_features):
        feature_stats = f"{feature_name}:\n  Mean: {stats['mean3d'][i]:.6f}\n  Variance: {stats['var3d'][i]:.6f}"
        output_stats.append(feature_stats)

    # Add timing and memory information
    end_time = time.time()
    duration = end_time - start_time
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_gb = memory_info.rss / (1024 ** 3)
    
    timing_info = [
        f"\nProcessing Information:",
        f"Calculation completed at: {datetime.now()}",
        f"Total duration: {duration/3600:.2f} hours ({duration/60:.2f} minutes)",
        f"Memory usage: {memory_gb:.2f} GB",
        f"Memory details:",
        f"  RSS (Resident Set Size): {memory_info.rss / (1024**2):.2f} MB",
        f"  VMS (Virtual Memory Size): {memory_info.vms / (1024**2):.2f} MB"
    ]
    output_stats.extend(timing_info)

    # Save to text file
    stats_txt_file = os.path.join(output_dir, txt_name)
    with open(stats_txt_file, 'w') as f:
        f.write('\n'.join(output_stats))
    
    print('\n'.join(output_stats), flush=True)
    print(f"\nDetailed statistics summary saved to: {stats_txt_file}", flush=True)

def main():
    args = parse_args()
    start_time = time.time()
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get file list
    pattern = 'ml_ecrad_ape_R2B05_myrunscript_1year_183min_tendencies_inputs_DOM01_ml_0001_lonlat_idx*_time_*.h5'
    files = sorted(glob.glob(os.path.join(args.input_dir, pattern)))
    total_files = len(files)
    print(f"Found {total_files} files")

    # Create dataset and dataloader with more conservative settings
    dataset = WeatherDataset(files)
    loader = DataLoader(
        dataset,
        batch_size=1,  # Process one sample at a time
        num_workers=args.num_workers,
        pin_memory=False,  # Set to True if using GPU
    )
    # Calculate statistics
    stats = calculate_statistics(loader, device, total_files)

    # Save results with specified file names
    save_statistics(stats, args.output_dir, start_time, args.pickle_name, args.txt_name)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()