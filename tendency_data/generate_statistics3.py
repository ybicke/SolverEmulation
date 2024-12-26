import h5py
import numpy as np
import glob
import os
import pickle
from datetime import datetime
import time

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Directory containing the input HDF5 files')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save the statistics')
    parser.add_argument('--pickle_name', type=str, default='normalizer_stats.pickle',
                      help='Name of the pickle file to save statistics')
    parser.add_argument('--txt_name', type=str, default='normalizer_stats_summary.txt',
                      help='Name of the text file to save summary')
    return parser.parse_args()

def update_welford_stats(existing_agg, new_values):
    """Update Welford's algorithm aggregates for computing mean and variance."""
    (count, mean, M2) = existing_agg
    
    for value in new_values:
        count += 1
        delta = value - mean
        mean += delta / count
        delta2 = value - mean
        M2 += delta * delta2
    
    return (count, mean, M2)

def print_and_save_summary(final_stats, duration, num_files, output_file):
    """Print and save statistics summary."""
    summary = [
        f"\nStatistics Summary (Generated on: {datetime.now()})",
        f"Processed {num_files} files in {duration/3600:.2f} hours",
        "\nTarget Variable (w):",
        f"  Mean: {final_stats['w']['mean'][0]:.6f}",
        f"  Std:  {final_stats['w']['std'][0]:.6f}",
        "\nSurface Features (x2d):"
    ]
    
    x2d_features = ['pres_sfc', 'cosmu0', 'qv_s']
    for i, feat in enumerate(x2d_features):
        summary.extend([
            f"  {feat}:",
            f"    Mean: {final_stats['x2d']['mean'][i]:.6f}",
            f"    Std:  {final_stats['x2d']['std'][i]:.6f}"
        ])
    
    summary.append("\n3D Features (x3d):")
    x3d_features = ['u', 'v', 'pres', 'geopot', 'qc', 'qi', 'qv', 'clc']
    for i, feat in enumerate(x3d_features):
        summary.extend([
            f"  {feat}:",
            f"    Mean: {final_stats['x3d']['mean'][i]:.6f}",
            f"    Std:  {final_stats['x3d']['std'][i]:.6f}"
        ])
    
    print('\n'.join(summary))
    with open(output_file, 'w') as f:
        f.write('\n'.join(summary))

def main():
    args = parse_args()
    
    start_time = time.time()
    print(f"Starting calculation at: {datetime.now()}")
    
    # Get list of all files
    pattern = 'ml_ecrad_ape_R2B05_*.h5'
    files = sorted(glob.glob(os.path.join(args.input_dir, pattern)))
    print(f"Found {len(files)} files")
    
    # Initialize statistics
    stats = {
        'w': (0, np.zeros(1), np.zeros(1)),
        'x2d': (0, np.zeros(3), np.zeros(3)),
        'x3d': (0, np.zeros(8), np.zeros(8))
    }
    
    # Process files
    for i, file_path in enumerate(files):
        # Print progress every 100 files
        if (i + 1) % 100 == 0:
            elapsed_time = time.time() - start_time
            print(f"Processed {i+1}/{len(files)} files. Elapsed time: {elapsed_time/3600:.2f} hours")
        
        with h5py.File(file_path, 'r') as f:
            # Process each dataset
            w_data = f['w'][:].reshape(-1)
            stats['w'] = update_welford_stats(stats['w'], w_data)
            
            x2d_data = f['x2d'][:].reshape(-1, 3)
            for j in range(3):
                stats['x2d'] = update_welford_stats(stats['x2d'], x2d_data[:, j])
            
            x3d_data = f['x3d'][:].reshape(-1, 8)
            for j in range(8):
                stats['x3d'] = update_welford_stats(stats['x3d'], x3d_data[:, j])
    
    # Calculate final statistics
    final_stats = {}
    for key in stats:
        count, mean, M2 = stats[key]
        variance = M2 / (count - 1)
        std = np.sqrt(variance)
        final_stats[key] = {'mean': mean, 'std': std}
    
    # Save statistics
    pickle_path = os.path.join(args.output_dir, args.pickle_name)
    txt_path = os.path.join(args.output_dir, args.txt_name)
    
    with open(pickle_path, 'wb') as f:
        pickle.dump(final_stats, f)
    
    # Print and save summary
    duration = time.time() - start_time
    print_and_save_summary(final_stats, duration, len(files), txt_path)

if __name__ == "__main__":
    main()