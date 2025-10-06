def calculate_output_statistics(input_dir, output_dir, features):
    import time
    import pickle
    import psutil
    import numpy as np
    import h5py
    import glob
    import os
    from datetime import datetime
    
    start_time = time.time()
    print(f"Starting calculation at: {datetime.now()}")

    # Adjust file search pattern as needed
    pattern = 'ml_ecrad_ape_R2B05_myrunscript_1year_183min_tendencies_DOM01_ml_0001_lonlat_idx*_time_*.h5'
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    total_files = len(files)
    print(f"Found {len(files)} files: {files[:5]}...")  # Show first few for debugging

    # Count total number of data points (per feature) for final division
    # e.g., each file y_data shape might be (81920, 70, num_features).
    # If each file has the same shape, we can detect it once from the first file:
    test_file = files[0] if files else None
    if not test_file:
        print("No files found. Exiting.")
        return

    with h5py.File(test_file, 'r') as ftest:
        shape = ftest['y'].shape  # e.g. (81920, 70, num_features)
    points_per_file = shape[0] * shape[1]  # e.g. 81920*70
    num_features = shape[-1]

    print(f"Detected shape: {shape}, implying {points_per_file} data points per feature, per file.")

    # Initialize accumulators for sums and sums of squares
    sum_ = np.zeros(num_features, dtype=np.float64)
    sumsq_ = np.zeros(num_features, dtype=np.float64)
    N = 0  # total number of files processed

    for file_idx, file_path in enumerate(files):
        with h5py.File(file_path, 'r') as f:
            y_data = f['y'][:]
            # y_data shape is presumably (81920, 70, num_features).
            # Flatten out the (81920, 70) part so that we have a 2D array [Npoints, features].
            # Then we can sum along axis=0 to get feature-wise sums.
            # Or do it carefully in parts to save memory, if needed.
            y_flat = y_data.reshape(-1, num_features)  # shape (81920*70, num_features)

            sum_ += y_flat.sum(axis=0)
            sumsq_ += (y_flat ** 2).sum(axis=0)

        N += 1

        # Progress print every 100 files:
        if (file_idx + 1) % 100 == 0:
            elapsed = (time.time() - start_time) / 60.0
            print(f"Processed {file_idx+1}/{total_files} files in {elapsed:.2f} minutes...")

    # Now compute total number of data points across all files
    total_points = N * points_per_file  # e.g. for N files, each with 81920*70 data points

    # Final mean and variance
    mean = sum_ / total_points
    # var = E(x^2) - [E(x)]^2
    # For an unbiased sample variance, we could do:
    # var = (sumsq_ - (sum_**2)/total_points) / (total_points - 1)
    # If you prefer population variance, you can omit the -1 in denominator.
    var = (sumsq_ - (sum_**2)/total_points) / (total_points - 1)

    final_stats = {
        'mean': mean,
        'var': var
    }

    # Save statistics
    output_file = os.path.join(output_dir, 'normalizer_stats_per_feat_y.pickle')
    with open(output_file, 'wb') as pf:
        pickle.dump(final_stats, pf)

    # Print a summary
    end_time = time.time()
    duration = end_time - start_time
    process = psutil.Process()
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / (1024**3)

    print("\n--- Feature-wise Statistics ---")
    for i, feat in enumerate(features):
        print(f"Feature: {feat}")
        print(f"  Mean: {mean[i]:.6f}")
        print(f"  Var:  {var[i]:.6f}\n")

    print("--- Processing Info ---")
    print(f"Total files processed: {N}")
    print(f"Total points: {total_points}")
    print(f"Total duration: {duration/60:.2f} minutes ({duration:.2f} sec)")
    print(f"Memory usage: {mem_gb:.2f} GB RSS")
    print(f"Final stats saved to: {output_file}")