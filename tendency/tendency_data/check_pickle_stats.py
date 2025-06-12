#!/usr/bin/env python3
"""
Simple script to check the contents of the statistics pickle file.
Shows the actual precision of stored variance values.
"""

import pickle
import numpy as np
import sys
import os

# Path to your pickle file - update this to your actual path
PICKLE_PATH = '/mydata/deepcloud/shared/h5_tendency_all/normalizer_stats_per_feat_y2_no_temp.pickle'

# Alternative paths to try
ALTERNATIVE_PATHS = [
    '/mydata/deepcloud/yves/h5_tendency_data_all/statistics/normalizer_stats_per_feat_y2.pickle',
    '/mydata/deepcloud/yves/h5_tendency_data_all/normalizer_stats_per_feat_y2_no_temp.pickle',
]

def check_pickle_stats(pickle_path):
    """Load and display statistics from pickle file."""
    
    # Check if file exists
    if not os.path.exists(pickle_path):
        print(f"File not found: {pickle_path}")
        return False
    
    print(f"\nChecking pickle file: {pickle_path}")
    print("=" * 80)
    
    # Load the pickle file
    with open(pickle_path, 'rb') as f:
        stats = pickle.load(f)
    
    # Check what's in the pickle
    print("\nKeys in pickle file:", list(stats.keys()))
    
    # Get the arrays
    means = stats['mean']
    variances = stats['var']
    
    print(f"\nNumber of features: {len(means)}")
    print(f"Data type of means: {type(means)}, dtype: {means.dtype if hasattr(means, 'dtype') else 'N/A'}")
    print(f"Data type of variances: {type(variances)}, dtype: {variances.dtype if hasattr(variances, 'dtype') else 'N/A'}")
    
    # Feature names (if you know them)
    feature_names = [
        'ddt_temp_sum',    # 0
        'temp',            # 1
        'ddt_temp_dyn',    # 2
        'ddt_u_sum',       # 3
        'ddt_v_sum',       # 4
        'ddt_qv_conv',     # 5
        'ddt_qc_conv',     # 6
        'ddt_qi_conv',     # 7
    ]
    
    # Display all values with proper scientific notation
    print("\n" + "=" * 80)
    print("FEATURE STATISTICS (showing actual stored precision):")
    print("=" * 80)
    
    for i in range(len(means)):
        feature_name = feature_names[i] if i < len(feature_names) else f"Feature_{i}"
        print(f"\n{i}: {feature_name}")
        print(f"   Mean:     {means[i]:+.15e}")
        print(f"   Variance: {variances[i]:+.15e}")
        print(f"   Std Dev:  {np.sqrt(variances[i]):+.15e}")
        
        # Check if variance is very small
        if variances[i] < 1e-10:
            print(f"   ⚠️  Very small variance detected!")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS:")
    print("=" * 80)
    
    # Find min/max variances
    non_zero_vars = variances[variances > 0]
    if len(non_zero_vars) > 0:
        min_var = non_zero_vars.min()
        max_var = variances.max()
        min_idx = np.where(variances == min_var)[0][0]
        max_idx = np.where(variances == max_var)[0][0]
        
        print(f"\nSmallest variance: {min_var:.15e} (feature {min_idx}: {feature_names[min_idx] if min_idx < len(feature_names) else 'Unknown'})")
        print(f"Largest variance:  {max_var:.15e} (feature {max_idx}: {feature_names[max_idx] if max_idx < len(feature_names) else 'Unknown'})")
        print(f"Variance ratio (max/min): {max_var/min_var:.2e}")
        
        # Calculate what the scales would be with k=4
        print(f"\nWith k=4 scaling:")
        print(f"Smallest scale: {4 * np.sqrt(min_var):.15e}")
        print(f"Largest scale:  {4 * np.sqrt(max_var):.15e}")
        
        # Check if any would be affected by different min_scale values
        print(f"\nChecking min_scale impact:")
        for min_scale in [1e-6, 1e-12, 1e-20, 1e-30]:
            affected = np.sum(4 * np.sqrt(variances) < min_scale)
            if affected > 0:
                print(f"  min_scale={min_scale:.0e}: {affected} features would be clamped")
            else:
                print(f"  min_scale={min_scale:.0e}: No features affected ✓")
    
    return True

def main():
    # Try the main path first
    if os.path.exists(PICKLE_PATH):
        check_pickle_stats(PICKLE_PATH)
    else:
        print(f"Primary path not found: {PICKLE_PATH}")
        print("\nTrying alternative paths...")
        
        found = False
        for alt_path in ALTERNATIVE_PATHS:
            if os.path.exists(alt_path):
                found = True
                check_pickle_stats(alt_path)
                break
        
        if not found:
            print("\nNo pickle files found. Please update the path in the script.")
            print("\nYou can also run this script with a path argument:")
            print(f"  python {sys.argv[0]} /path/to/your/pickle/file.pickle")

if __name__ == "__main__":
    # Allow passing pickle path as command line argument
    if len(sys.argv) > 1:
        check_pickle_stats(sys.argv[1])
    else:
        main() 