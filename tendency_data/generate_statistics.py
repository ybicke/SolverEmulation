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
    total_files = len(files)
    print(f"Found {len(files)} files")
    print(f"Starting processing of {total_files} files at: {datetime.now()}", flush=True)

    # Initialize statistics dictionaries for each feature
    stats = {
        'count': 0,
        'mean_w': None,   # Will be shape (1,) for w
        'M2_w': None,     # Will be shape (1,) for w
        'mean2d': None,   # Will be shape (3,) for 3 features
        'M2_2d': None,    # Will be shape (3,) for 3 features
        'mean3d': None,   # Will be shape (8,) for 8 features
        'M2_3d': None,    # Will be shape (8,) for 8 features
    }


    
    for file_idx, file_path in enumerate(files):  # Removed tqdm temporarily
        with h5py.File(file_path, 'r') as f:
            
       
            
            w_data = f['w'][:].reshape(-1, 1)
            x2d_data = f['x2d'][:]
            x3d_data = f['x3d'][:].reshape(-1, 8)

            if stats['mean_w'] is None:
                stats['mean_w'] = np.zeros(1)
                stats['M2_w'] = np.zeros(1)
                stats['mean2d'] = np.zeros(3)
                stats['M2_2d'] = np.zeros(3)
                stats['mean3d'] = np.zeros(8)
                stats['M2_3d'] = np.zeros(8)
            
            stats['count'] += 1

            # Update w statistics
            delta = w_data - stats['mean_w']
            stats['mean_w'] += np.mean(delta, axis=0)
            delta2 = w_data - stats['mean_w']
            stats['M2_w'] += np.sum(delta * delta2, axis=0)

            # Update x2d statistics
            for i in range(3):
                feature_data = x2d_data[:, i]
                delta = feature_data - stats['mean2d'][i]
                stats['mean2d'][i] += np.mean(delta)
                delta2 = feature_data - stats['mean2d'][i]
                stats['M2_2d'][i] += np.sum(delta * delta2)

            # Update x3d statistics
            for j in range(8):
                feature_data = x3d_data[:, j]
                delta = feature_data - stats['mean3d'][j]
                stats['mean3d'][j] += np.mean(delta)
                delta2 = feature_data - stats['mean3d'][j]
                stats['M2_3d'][j] += np.sum(delta * delta2)


                # Print progress every 200 files
        if (file_idx + 1) % 100 == 0:
            progress = ((file_idx + 1) / total_files) * 100
            elapsed_time = time.time() - start_time
            current_time = datetime.now()
            print(f"\nProgress update at {current_time}:", flush=True)
            print(f"Processed: {file_idx+1}/{total_files} files ({progress:.2f}%)", flush=True)
            print(f"Time elapsed: {elapsed_time/3600:.2f} hours", flush=True)
            print(f"Current file: {os.path.basename(file_path)}", flush=True)
            print("-" * 50, flush=True)    

    # Calculate final statistics
    final_stats = {
        'mean_w': stats['mean_w'],
        'var_w': stats['M2_w'] / (stats['count'] * 81920 * 71 - 1),
        'mean2d': stats['mean2d'],
        'var2d': stats['M2_2d'] / (stats['count'] * 81920 - 1),
        'mean3d': stats['mean3d'],
        'var3d': stats['M2_3d'] / (stats['count'] * 81920 * 70 - 1)
    }

    # Save statistics
    output_file = os.path.join(output_dir, 'normalizer_stats_per_feat_statistics1.pickle')
    with open(output_file, 'wb') as f:
        pickle.dump(final_stats, f)

    
    # Print and save feature-wise statistics with physical meanings
    output_stats = []  # List to collect all statistics for text file
    
    # Prepare header and timestamp
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
        # Print timing and memory information
    end_time = time.time()
    duration = end_time - start_time
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
    stats_txt_file = os.path.join(output_dir, 'normalizer_stats_per_feat_statistics1.txt')
    with open(stats_txt_file, 'w') as f:
        f.write('\n'.join(output_stats))
    
    print(f"\nDetailed statistics summary saved to: {stats_txt_file}")

if __name__ == "__main__":
    args = parse_args()
    calculate_dataset_statistics(args.input_dir, args.output_dir)