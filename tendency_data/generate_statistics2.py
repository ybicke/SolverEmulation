"""I went for Welford's algorithm beacause it is generally preferred for its numerical stability, especially when dealing with large datasets or
when high precision is required. The CIFAR10 approach might be simpler to implement and understand."""


import h5py
import numpy as np
import glob
import os
import pickle
from tqdm import tqdm
import argparse
from datetime import datetime
import time
import psutil



    

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate dataset statistics')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Directory containing the input h5 files')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save the statistics file')
    return parser.parse_args()

def calculate_dataset_statistics(input_dir, output_dir):
    start_time = time.time()
    print(f"Starting calculation at: {datetime.now()}")
    
    # Pattern modified to match only files with idx0000 for debugging
    pattern = 'ml_ecrad_ape_R2B05_myrunscript_1year_183min_tendencies_inputs_DOM01_ml_0001_lonlat_idx*_time_*.h5'
    
    # Get list of all files
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    print(f"Found {len(files)} files")

    # Initialize statistics dictionaries for each feature
    stats = {
        'N_w': 0,
        'mean_w': 0.0,
        'M2_w': 0.0,
        'N2d': 0,
        'mean2d': np.zeros(3),
        'M2_2d': np.zeros(3),
        'N3d': 0,
        'mean3d': np.zeros(8),
        'M2_3d': np.zeros(8),
    }
    
    
    """
    delta = new_value - mean_old     # difference between new value and old mean
    mean_new = mean_old + delta/count # update mean
    delta2 = new_value - mean_new    # difference between new value and new mean
    M2 += delta * delta2             # update M2
    
    
    delta * delta2 = (new_value - mean_old) * (new_value - mean_new)
                = (new_value - mean_old) * (new_value - (mean_old + delta/count))
                = (new_value - mean_old) * (new_value - mean_old - (new_value - mean_old)/count)
                = (new_value - mean_old)^2 * (1 - 1/count)
    """

    # Process each chunk file
    for chunk_file in tqdm(files, desc="Processing chunks"):
        with h5py.File(chunk_file, 'r') as f:
            # Load data from the chunk
            # w_data shape: (num_samples_w,)
            w_data = f['w'][:].reshape(-1)
            
            # x2d_data shape: (num_samples_2d, 3)
            x2d_data = f['x2d'][:]
            
            # x3d_data shape: (num_samples_3d, 8)
            x3d_data = f['x3d'][:].reshape(-1, 8)
        
        # --- Process w_data (target variable) ---
        n_batch_w = w_data.shape[0]
        batch_mean_w = np.mean(w_data, axis=0)
        batch_M2_w = np.sum((w_data - batch_mean_w) ** 2, axis=0)
        
        delta_w = batch_mean_w - stats['mean_w']
        N_total_w = stats['N_w'] + n_batch_w
        stats['mean_w'] += delta_w * (n_batch_w / N_total_w)
        stats['M2_w'] += batch_M2_w + delta_w ** 2 * stats['N_w'] * n_batch_w / N_total_w
        stats['N_w'] = N_total_w
        
        # --- Process x2d_data (2D features) ---
        n_batch_2d = x2d_data.shape[0]
        batch_mean_2d = np.mean(x2d_data, axis=0)
        batch_M2_2d = np.sum((x2d_data - batch_mean_2d) ** 2, axis=0)
        
        delta_2d = batch_mean_2d - stats['mean2d']
        N_total_2d = stats['N2d'] + n_batch_2d
        stats['mean2d'] += delta_2d * (n_batch_2d / N_total_2d)
        stats['M2_2d'] += batch_M2_2d + (delta_2d ** 2) * stats['N2d'] * n_batch_2d / N_total_2d
        stats['N2d'] = N_total_2d
        
        # --- Process x3d_data (3D features) ---
        n_batch_3d = x3d_data.shape[0]
        batch_mean_3d = np.mean(x3d_data, axis=0)
        batch_M2_3d = np.sum((x3d_data - batch_mean_3d) ** 2, axis=0)
        
        delta_3d = batch_mean_3d - stats['mean3d']
        N_total_3d = stats['N3d'] + n_batch_3d
        stats['mean3d'] += delta_3d * (n_batch_3d / N_total_3d)
        stats['M2_3d'] += batch_M2_3d + (delta_3d ** 2) * stats['N3d'] * n_batch_3d / N_total_3d
        stats['N3d'] = N_total_3d
    
    # Calculate final statistics
    final_stats = {
        'mean_w': np.array([stats['mean_w']]),
        'var_w': np.array([stats['M2_w'] / (stats['N_w'] - 1)]) if stats['N_w'] > 1 else np.array([0.0]),
        'mean2d': stats['mean2d'],
        'var2d': stats['M2_2d'] / (stats['N2d'] - 1),
        'mean3d': stats['mean3d'],
        'var3d': stats['M2_3d'] / (stats['N3d'] - 1)
    }
    
    
    # Save statistics
    output_file = os.path.join(output_dir, 'normalizer_stats_per_feat.pickle')
    with open(output_file, 'wb') as f:
        pickle.dump(final_stats, f)

    # Print timing and memory information
    end_time = time.time()
    duration = end_time - start_time
  
    
    # Print and save feature-wise statistics with physical meanings
    output_stats = []  # List to collect all statistics for text file
    header = f"\nFeature-wise statistics summary"
    timestamp = f"Generated on: {datetime.now()}"
    output_stats.extend([header, timestamp, ""])
    
    # w statistics (target variable)
    w_stats = "Target Variable:"
    w_details = f"w (Vertical velocity) [m/s]:\n  Mean: {final_stats['mean_w'][0]:.6f}\n  Variance: {final_stats['var_w'][0]:.6f}"
    output_stats.extend([w_stats, w_details, ""])
    
    # x2d statistics
    x2d_stats = "Surface-level Features (x2d):"
    output_stats.append(x2d_stats)
    x2d_features = ['pres_sfc - Surface pressure [hPa]', 
                    'cosmu0 - Cosine of solar zenith angle [-]', 
                    'qv_s - Surface water vapor specific humidity [kg/kg]']
    
    for i, feature_name in enumerate(x2d_features):
        feature_stats = f"{feature_name}:\n  Mean: {final_stats['mean2d'][i]:.6f}\n  Variance: {final_stats['var2d'][i]:.6f}"
        output_stats.append(feature_stats)
    output_stats.append("")
    
    # x3d statistics
    x3d_stats = "3D Features (x3d):"
    output_stats.append(x3d_stats)
    x3d_features = ['u - Zonal wind [m/s]',
                    'v - Meridional wind [m/s]',
                    'pres - Pressure [hPa]',
                    'geopot - Geopotential [m²/s²]',
                    'qc - Cloud water content [kg/kg]',
                    'qi - Cloud ice content [kg/kg]',
                    'qv - Water vapor specific humidity [kg/kg]',
                    'clc - Cloud cover fraction [-]']
    
    for i, feature_name in enumerate(x3d_features):
        feature_stats = f"{feature_name}:\n  Mean: {final_stats['mean3d'][i]:.6f}\n  Variance: {final_stats['var3d'][i]:.6f}"
        output_stats.append(feature_stats)

    # Add timing and memory information
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_gb = memory_info.rss / (1024 * 1024 * 1024)  # Convert bytes to GB
    
    timing_info = [
        f"\nProcessing Information:",
        f"Calculation completed at: {datetime.now()}",
        f"Total duration: {duration/3600:.2f} hours ({duration/60:.2f} minutes)",
        f"Number of files processed: {stats['count']}",
        f"Memory usage: {memory_gb:.2f} GB",
        f"Memory details:",
        f"  RSS (Resident Set Size): {memory_info.rss / (1024*1024):.2f} MB",
        f"  VMS (Virtual Memory Size): {memory_info.vms / (1024*1024):.2f} MB"
    ]
    output_stats.extend(timing_info)

    # Print to console
    print('\n'.join(output_stats))
    
    # Save to text file
    stats_txt_file = os.path.join(output_dir, 'normalizer_stats_summary.txt')
    with open(stats_txt_file, 'w') as f:
        f.write('\n'.join(output_stats))
    
    print(f"\nDetailed statistics summary saved to: {stats_txt_file}")

if __name__ == "__main__":
    args = parse_args()
    calculate_dataset_statistics(args.input_dir, args.output_dir)