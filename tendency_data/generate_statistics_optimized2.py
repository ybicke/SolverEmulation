import h5py
import numpy as np
import os
import glob
from datetime import datetime
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate dataset statistics')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing the input H5 files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the statistics file')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes')
    parser.add_argument('--pickle_name', type=str, default='normalizer_stats_per_feat_optimized.pickle',
                        help='Name of the pickle output file')
    parser.add_argument('--txt_name', type=str, default='normalizer_stats_summary_optimized.txt',
                        help='Name of the text output file')
    return parser.parse_args()

def process_file(filename):
    try:
        with h5py.File(filename, 'r') as h:
            w = h['w'][:]      # Shape: (81920, 71, 1)
            x2d = h['x2d'][:]  # Shape: (81920, 3)
            x3d = h['x3d'][:]  # Shape: (81920, 70, 8)

        # Flatten arrays and compute sums and sums of squares
        w_flat = w.flatten()
        w_sum = np.sum(w_flat, dtype=np.float64)
        w_sq_sum = np.sum(w_flat ** 2, dtype=np.float64)
        w_count = w_flat.size

        x2d_flat = x2d.reshape(-1, x2d.shape[-1])
        x2d_sum = np.sum(x2d_flat, axis=0, dtype=np.float64)
        x2d_sq_sum = np.sum(x2d_flat ** 2, axis=0, dtype=np.float64)
        x2d_count = x2d_flat.shape[0]

        x3d_flat = x3d.reshape(-1, x3d.shape[-1])
        x3d_sum = np.sum(x3d_flat, axis=0, dtype=np.float64)
        x3d_sq_sum = np.sum(x3d_flat ** 2, axis=0, dtype=np.float64)
        x3d_count = x3d_flat.shape[0]

        result = {
            'w_sum': w_sum,
            'w_sq_sum': w_sq_sum,
            'w_count': w_count,

            'x2d_sum': x2d_sum,
            'x2d_sq_sum': x2d_sq_sum,
            'x2d_count': x2d_count,

            'x3d_sum': x3d_sum,
            'x3d_sq_sum': x3d_sq_sum,
            'x3d_count': x3d_count,
        }
        return result
    except Exception as e:
        print(f"Error processing file {filename}: {e}")
        return None

def save_statistics(stats, output_dir, start_time, pickle_name, txt_name):
    # Save raw statistics
    output_file = os.path.join(output_dir, pickle_name)
    with open(output_file, 'wb') as f:
        pickle.dump(stats, f)

    # Prepare human-readable output
    output_stats = []

    # Header
    header = f"\nFeature-wise Statistics Summary"
    timestamp = f"Generated on: {datetime.now()}"
    output_stats.extend([header, timestamp, ""])

    # w statistics
    w_stats = "Target Variable:"
    w_details = f"w (Vertical velocity) [m/s]:\n  Mean: {stats['mean_w']:.6f}\n  Variance: {stats['var_w']:.6f}"
    output_stats.extend([w_stats, w_details, ""])

    # x2d statistics
    x2d_stats = "Surface-level Features (x2d):"
    output_stats.append(x2d_stats)
    x2d_features = [
        'pres_sfc - Surface pressure [Pa]',
        'cosmu0 - Cosine of solar zenith angle [-]',
        'qv_s - Surface water vapor specific humidity [kg/kg]'
    ]

    for i, feature_name in enumerate(x2d_features):
        feature_stats = f"{feature_name}:\n  Mean: {stats['mean2d'][i]:.6f}\n  Variance: {stats['var2d'][i]:.6f}"
        output_stats.append(feature_stats)
    output_stats.append("")

    # x3d statistics
    x3d_stats = "3D Features (x3d):"
    output_stats.append(x3d_stats)
    x3d_features = [
        'u - Zonal wind [m/s]',
        'v - Meridional wind [m/s]',
        'pres - Pressure [Pa]',
        'geopot - Geopotential [m²/s²]',
        'qc - Cloud water content [kg/kg]',
        'qi - Cloud ice content [kg/kg]',
        'qv - Water vapor specific humidity [kg/kg]',
        'clc - Cloud cover fraction [-]'
    ]

    for i, feature_name in enumerate(x3d_features):
        feature_stats = f"{feature_name}:\n  Mean: {stats['mean3d'][i]:.6f}\n  Variance: {stats['var3d'][i]:.6f}"
        output_stats.append(feature_stats)

    # Add timing information
    end_time = datetime.now()
    duration = end_time - start_time

    timing_info = [
        f"\nProcessing Information:",
        f"Calculation completed at: {end_time}",
        f"Total duration: {duration}",
    ]
    output_stats.extend(timing_info)

    # Save to text file
    stats_txt_file = os.path.join(output_dir, txt_name)
    with open(stats_txt_file, 'w') as f:
        f.write('\n'.join(output_stats))

    print('\n'.join(output_stats))
    print(f"\nDetailed statistics summary saved to: {stats_txt_file}")

def main():
    args = parse_args()
    start_time = datetime.now()
    print(f"Processing started at {start_time}")

    # Get file list
    pattern = 'ml_ecrad_ape_R2B05_myrunscript_1year_183min_tendencies_inputs_DOM01_ml_0001_lonlat_idx*_time_*.h5'
    files = sorted(glob.glob(os.path.join(args.input_dir, pattern)))
    total_files = len(files)
    print(f"Found {total_files} files")

    # Initialize accumulators
    total_w_sum = 0.0
    total_w_sq_sum = 0.0
    total_w_count = 0

    total_x2d_sum = np.zeros(3, dtype=np.float64)
    total_x2d_sq_sum = np.zeros(3, dtype=np.float64)
    total_x2d_count = 0

    total_x3d_sum = np.zeros(8, dtype=np.float64)
    total_x3d_sq_sum = np.zeros(8, dtype=np.float64)
    total_x3d_count = 0

    timesteps_processed = 0
    timesteps_per_update = 200

    # Use ProcessPoolExecutor to process files in parallel
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_file, f): f for f in files}

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result is None:
                continue  # Skip files that failed to process

            # Update accumulators
            total_w_sum += result['w_sum']
            total_w_sq_sum += result['w_sq_sum']
            total_w_count += result['w_count']

            total_x2d_sum += result['x2d_sum']
            total_x2d_sq_sum += result['x2d_sq_sum']
            total_x2d_count += result['x2d_count']

            total_x3d_sum += result['x3d_sum']
            total_x3d_sq_sum += result['x3d_sq_sum']
            total_x3d_count += result['x3d_count']

            timesteps_processed += 1

            if timesteps_processed % timesteps_per_update == 0 or timesteps_processed == total_files:
                progress = (timesteps_processed / total_files) * 100
                current_time = datetime.now()
                elapsed_time = current_time - start_time
                print(f"\nProgress update at {current_time}:")
                print(f"Processed: {timesteps_processed}/{total_files} timesteps ({progress:.2f}%)")
                print(f"Elapsed time: {elapsed_time}")

    # Calculate final statistics
    mean_w = total_w_sum / total_w_count
    var_w = (total_w_sq_sum / total_w_count) - (mean_w ** 2)

    mean_2d = total_x2d_sum / total_x2d_count
    var_2d = (total_x2d_sq_sum / total_x2d_count) - (mean_2d ** 2)

    mean_3d = total_x3d_sum / total_x3d_count
    var_3d = (total_x3d_sq_sum / total_x3d_count) - (mean_3d ** 2)

    # Prepare statistics
    stats = {
        'mean_w': mean_w,          # Scalar
        'var_w': var_w,            # Scalar
        'mean2d': mean_2d,         # Array of shape (3,)
        'var2d': var_2d,           # Array of shape (3,)
        'mean3d': mean_3d,         # Array of shape (8,)
        'var3d': var_3d            # Array of shape (8,)
    }

    # Save results
    save_statistics(stats, args.output_dir, start_time, args.pickle_name, args.txt_name)

    end_time = datetime.now()
    print(f"Processing completed at {end_time}")
    print(f"Total duration: {end_time - start_time}")

if __name__ == "__main__":
    main()