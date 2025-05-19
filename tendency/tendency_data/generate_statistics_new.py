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

    # Adjust the glob pattern as needed
    pattern = 'ml_ecrad_ape_R2B05_myrunscript_1year_183min_tendencies_inputs_DOM01_ml_0001_lonlat_idx*_time_*.h5'
    
    # Get list of all files
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    total_files = len(files)
    print(f"Found {len(files)} files")
    print(f"Starting processing of {total_files} files at: {datetime.now()}", flush=True)

    # Initialize statistics dictionaries for each feature
    stats = {
        'count': 0,
        'mean_w': None,   # shape (1,) for w
        'M2_w': None,     # shape (1,) for w
        'mean2d': None,   # shape (3,) for x2d (pres_sfc, cosmu0, qv_s)
        'M2_2d': None,    # shape (3,)
        'mean3d': None,   # shape (10,) for x3d
        'M2_3d': None,    # shape (10,)
    }

    for file_idx, file_path in enumerate(files):  # You can re-add tqdm if desired
        with h5py.File(file_path, 'r') as f:
            # Read the w data
            w_data = f['w'][:].reshape(-1, 1)  # single feature target

            # Read the 2D features (3 total features)
            x2d_data = f['x2d'][:]
            # shape might be (N, 3) already, so no reshape needed, or adapt if your data shape is different

            # Read the 3D features (now 10 total features)
            x3d_data = f['x3d'][:].reshape(-1, 10)

            # Initialize arrays if they are None
            if stats['mean_w'] is None:
                stats['mean_w']   = np.zeros(1)
                stats['M2_w']     = np.zeros(1)
                stats['mean2d']   = np.zeros(3)
                stats['M2_2d']    = np.zeros(3)
                stats['mean3d']   = np.zeros(10)
                stats['M2_3d']    = np.zeros(10)
            
            # Increment the count of processed files
            stats['count'] += 1

            # -------------------
            # Update w statistics
            # -------------------
            delta = w_data - stats['mean_w']
            stats['mean_w'] += np.mean(delta, axis=0)
            delta2 = w_data - stats['mean_w']
            stats['M2_w'] += np.sum(delta * delta2, axis=0)

            # ------------------------
            # Update x2d statistics
            # ------------------------
            for i in range(3):
                feature_data = x2d_data[:, i]
                delta = feature_data - stats['mean2d'][i]
                stats['mean2d'][i] += np.mean(delta)
                delta2 = feature_data - stats['mean2d'][i]
                stats['M2_2d'][i] += np.sum(delta * delta2)

            # ------------------------
            # Update x3d statistics
            # ------------------------
            for j in range(10):
                feature_data = x3d_data[:, j]
                delta = feature_data - stats['mean3d'][j]
                stats['mean3d'][j] += np.mean(delta)
                delta2 = feature_data - stats['mean3d'][j]
                stats['M2_3d'][j] += np.sum(delta * delta2)

        # Print progress every 100 files
        if (file_idx + 1) % 100 == 0:
            progress = ((file_idx + 1) / total_files) * 100
            elapsed_time = time.time() - start_time
            current_time = datetime.now()
            print(f"\nProgress update at {current_time}:", flush=True)
            print(f"Processed: {file_idx + 1}/{total_files} files ({progress:.2f}%)", flush=True)
            print(f"Time elapsed: {elapsed_time/3600:.2f} hours", flush=True)
            print(f"Current file: {os.path.basename(file_path)}", flush=True)
            print("-" * 50, flush=True)

    # ---------------------------------------
    # Calculate final means and variances
    # ---------------------------------------
    # Note: The denominators below are placeholders. 
    # They depend on how many total samples exist in each file 
    # (N * number_of_vertical_levels). Adjust accordingly.
    
    N_w  = (stats['count'] * 81920 * 71)  # <--- Example; adapt for your actual data dimension
    N_2d = (stats['count'] * 81920)       # <--- Example
    N_3d = (stats['count'] * 81920 * 70)  # <--- Example

    final_stats = {
        'mean_w': stats['mean_w'],
        'var_w':  stats['M2_w']   / (N_w - 1),
        'mean2d': stats['mean2d'],
        'var2d':  stats['M2_2d']  / (N_2d - 1),
        'mean3d': stats['mean3d'],
        'var3d':  stats['M2_3d']  / (N_3d - 1),
    }

    # Save statistics as a pickle
    output_file = os.path.join(output_dir, 'normalizer_stats_per_feat_statistics_new.pickle')
    with open(output_file, 'wb') as f:
        pickle.dump(final_stats, f)

    # -----------------------------------------
    # Prepare a human-readable text summary
    # -----------------------------------------
    output_stats = []
    header = f"\nFeature-wise statistics summary"
    timestamp = f"Generated on: {datetime.now()}"
    output_stats.extend([header, timestamp, ""])

    # w statistic
    w_stats = "Target Variable:"
    w_details = (f"w (Vertical velocity) [m/s]:\n"
                 f"  Mean: {final_stats['mean_w'][0]:.6f}\n"
                 f"  Variance: {final_stats['var_w'][0]:.6f}")
    output_stats.extend([w_stats, w_details, ""])

    # x2d statistics
    x2d_stats = "Surface-level Features (x2d):"
    output_stats.append(x2d_stats)
    x2d_features = [
        'pres_sfc - Surface pressure [hPa]', 
        'cosmu0 - Cosine of solar zenith angle [-]', 
        'qv_s - Surface water vapor specific humidity [kg/kg]'
    ]
    for i, feature_name in enumerate(x2d_features):
        feature_stats = (f"{feature_name}:\n"
                         f"  Mean: {final_stats['mean2d'][i]:.6f}\n"
                         f"  Variance: {final_stats['var2d'][i]:.6f}")
        output_stats.append(feature_stats)
    output_stats.append("")

    # x3d statistics (10 features)
    x3d_stats = "3D Features (x3d):"
    output_stats.append(x3d_stats)
    x3d_features = [
        'u - Zonal wind [m/s]',
        'v - Meridional wind [m/s]',
        'w_interpolated [m/s]',
        'pres - Pressure [hPa]',
        'geopot - Geopotential [m^2/s^2]',
        'clc - Cloud cover fraction [-]',
        'qc - Cloud water content [kg/kg]',
        'qi - Cloud ice content [kg/kg]',
        'qv - Water vapor specific humidity [kg/kg]',
        'temp - Temperature [K]'
    ]
    for i, feature_name in enumerate(x3d_features):
        feature_stats = (f"{feature_name}:\n"
                         f"  Mean: {final_stats['mean3d'][i]:.6f}\n"
                         f"  Variance: {final_stats['var3d'][i]:.6f}")
        output_stats.append(feature_stats)

    # Add timing and memory usage
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

    # Print summary to console
    print('\n'.join(output_stats))

    # Save summary to text file
    stats_txt_file = os.path.join(output_dir, 'normalizer_stats_per_feat_statistics_new.txt')
    with open(stats_txt_file, 'w') as f:
        f.write('\n'.join(output_stats))

    print(f"\nDetailed statistics summary saved to: {stats_txt_file}")

if __name__ == "__main__":
    args = parse_args()
    calculate_dataset_statistics(args.input_dir, args.output_dir)
