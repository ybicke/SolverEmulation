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
    parser = argparse.ArgumentParser(description='Calculate dataset statistics for output variables')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing the input h5 files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the statistics file')
    parser.add_argument('-f', '--features', nargs='+',
                        default=[
                            'ddt_temp_sum',    # sum of temperature tendencies
                            'temp',            # Temperature
                            'ddt_temp_dyn',    # dynamical temperature tendency
                            'ddt_u_sum',       # sum of zonal wind tendencies
                            'ddt_v_sum',       # sum of meridional wind tendencies
                            'ddt_qv_conv',     # convective tendency of absolute humidity
                            'ddt_qc_conv',     # convective tendency of cloud water mass density
                            'ddt_qi_conv',     # convective tendency of cloud ice mass density
                        ])
    return parser.parse_args()

def calculate_output_statistics(input_dir, output_dir, features):
    start_time = time.time()
    print(f"Starting calculation at: {datetime.now()}")
    
    # Pattern modified to match only files with idx0000 for debugging
    pattern = 'ml_ecrad_ape_R2B05_myrunscript_1year_183min_tendencies_DOM01_ml_0001_lonlat_idx*_time_*.h5'
    
    # Get list of all files
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    print(f"Found {len(files)} files")

    # Initialize statistics dictionaries for each feature
    stats = {
        'N': 0,
        'mean': np.zeros(len(features)),
        'M2': np.zeros(len(features)),
    }

    # Process each chunk file
    for chunk_file in tqdm(files, desc="Processing chunks"):
        with h5py.File(chunk_file, 'r') as f:
            # Load data from the chunk
            # y_data shape: (num_samples, num_features)
            
            y_data = f['y'][:].reshape(-1, len(features))
            
                    
        # Process y_data (output variables)
        n_batch = y_data.shape[0]
        batch_mean = np.mean(y_data, axis=0)
        batch_M2 = np.sum((y_data - batch_mean) ** 2, axis=0)
        
        delta = batch_mean - stats['mean']
        N_total = stats['N'] + n_batch
        stats['mean'] += delta * (n_batch / N_total)
        stats['M2'] += batch_M2 + (delta ** 2) * stats['N'] * n_batch / N_total
        stats['N'] = N_total
    
    # Calculate final statistics
    final_stats = {
        'mean': stats['mean'],
        'var': stats['M2'] / (stats['N'] - 1)
    }
    
    # Save statistics
    output_file = os.path.join(output_dir, 'normalizer_stats_per_feat_y2.pickle')
    with open(output_file, 'wb') as f:
        pickle.dump(final_stats, f)

    # Print timing and memory information
    end_time = time.time()
    duration = end_time - start_time
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_gb = memory_info.rss / (1024 * 1024 * 1024)  # Convert bytes to GB
    
    # Print and save feature-wise statistics with physical meanings
    output_stats = []  # List to collect all statistics for text file
    header = f"\nFeature-wise statistics summary for output variables"
    timestamp = f"Generated on: {datetime.now()}"
    output_stats.extend([header, timestamp, ""])
    
    # Output variable statistics
    output_stats.append("Output Variables:")
    for i, feature_name in enumerate(features):
        feature_stats = f"{feature_name}:\n  Mean: {final_stats['mean'][i]:.6f}\n  Variance: {final_stats['var'][i]:.6f}"
        output_stats.append(feature_stats)
    output_stats.append("")

    # Add timing and memory information
    timing_info = [
        f"\nProcessing Information:",
        f"Calculation completed at: {datetime.now()}",
        f"Total duration: {duration/3600:.2f} hours ({duration/60:.2f} minutes)",
        f"Number of files processed: {len(files)}",
        f"Memory usage: {memory_gb:.2f} GB",
        f"Memory details:",
        f"  RSS (Resident Set Size): {memory_info.rss / (1024*1024):.2f} MB",
        f"  VMS (Virtual Memory Size): {memory_info.vms / (1024*1024):.2f} MB"
    ]
    output_stats.extend(timing_info)

    # Print to console
    print('\n'.join(output_stats))
    
    # Save to text file
    stats_txt_file = os.path.join(output_dir, 'normalizer_stats_per_feat_y2.txt')
    with open(stats_txt_file, 'w') as f:
        f.write('\n'.join(output_stats))
    
    print(f"\nDetailed statistics summary saved to: {stats_txt_file}")

if __name__ == "__main__":
    args = parse_args()
    calculate_output_statistics(args.input_dir, args.output_dir, args.features)