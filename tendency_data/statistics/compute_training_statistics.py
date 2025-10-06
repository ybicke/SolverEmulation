#!/usr/bin/env python3
"""
Compute training set statistics (mean and variance) for tendency predictions.
Uses the same data loading pipeline as training to ensure consistency.
Implements Welford's online algorithm for memory-efficient computation.
"""

import os
import re
import sys
import time
import glob
import pickle
import logging
import random
import argparse
from os.path import join, dirname

import torch
import numpy as np
from torch.utils.data import DataLoader

# Import the same data loading functionality as training
from data_loaders_tendency import IconColumnIterableDataset

# Set up logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Reproducible results
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compute training set statistics for tendency prediction.')
    
    # Dataset paths
    parser.add_argument('--dataset_input', type=str, required=True,
                       help='Path to the input dataset directory')
    parser.add_argument('--dataset_output', type=str, required=True,
                       help='Path to the output dataset directory')
    parser.add_argument('--save_path', type=str, required=True,
                       help='Path to save the computed statistics')
    
    # Data selection parameters (same as training script)
    parser.add_argument('--percent', type=float, default=1.0,
                       help='Percentage of data to use (default: 1.0 = 100%%)')
    parser.add_argument('--subsample', type=float, default=1.0,
                       help='Subsampling rate within files (default: 1.0 = 100%%)')
    
    # Data loading parameters
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for data loading')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of workers for data loading')
    parser.add_argument('--prefetch-factor', type=int, default=2,
                       help='Prefetch factor for data loading')
    
    # Statistics options
    parser.add_argument('--compute-variance', action='store_true',
                       help='Compute both mean and variance (default: mean only)')
    
    return parser.parse_args()


def get_sorted_file_lists(dataset_input, dataset_output):
    """Get sorted input and output file lists."""
    logger.info('Finding and sorting files...')
    
    input_files = glob.glob(join(dataset_input, '*_inputs_*.h5'))
    output_files = glob.glob(join(dataset_output, '*_tendencies_*.h5'))
    
    logger.info(f'Found {len(input_files)} input files and {len(output_files)} output files')
    
    if not input_files:
        raise ValueError(f'No input files found in {dataset_input}')
    if not output_files:
        raise ValueError(f'No output files found in {dataset_output}')
    
    # Extract and sort by time indices
    def get_time_index(filename):
        return float(re.search(r'_time_(\d+\.\d+)\.h5', filename).group(1))
    
    input_files.sort(key=get_time_index)
    output_files.sort(key=get_time_index)
    
    # Verify time indices match
    input_times = [get_time_index(f) for f in input_files]
    output_times = [get_time_index(f) for f in output_files]
    
    if input_times != output_times:
        raise ValueError("Input and output files have mismatched time indices")
    
    return input_files, output_files


def select_training_files(input_files, output_files, percent=1.0):
    """Select training files using the same logic as the training script."""
    # Use indices 200:2000 for training (same as training script)
    train_input = input_files[200:2000]
    train_output = output_files[200:2000]
    
    logger.info(f'Selected {len(train_input)} training files (indices 200:2000)')
    
    # Apply percentage sampling if requested
    if percent < 1.0:
        n_files = len(train_input)
        n_keep = max(1, int(percent * n_files))
        indices = np.random.RandomState(SEED).choice(n_files, n_keep, replace=False)
        
        train_input = [train_input[i] for i in indices]
        train_output = [train_output[i] for i in indices]
        logger.info(f'Subsampled to {len(train_input)} files ({percent*100:.1f}%)')
    
    return train_input, train_output


def compute_statistics_welford(data_loader, compute_variance=False):
    """
    Compute mean (and optionally variance) using Welford's online algorithm.
    Memory efficient - processes data incrementally.
    
    Args:
        data_loader: PyTorch DataLoader yielding batches of (input, coords, target, mask)
        compute_variance: Whether to compute variance in addition to mean
        
    Returns:
        mean: Tensor of shape [n_features] containing feature-wise means
        variance: (if compute_variance=True) Tensor of shape [n_features] containing variances
    """
    logger.info(f'Computing statistics (variance={compute_variance})...')
    
    count = 0
    mean = None
    M2 = None  # For variance computation
    
    start_time = time.time()
    
    for batch_idx, batch_data in enumerate(data_loader):
        # Extract target data - format is (input, coords, target, mask)
        if len(batch_data) == 4:
            _, _, batch_y, _ = batch_data
        else:
            # Fallback for different data formats
            batch_y = batch_data[2] if len(batch_data) > 2 else batch_data[1]
        
        # Move to CPU to save GPU memory
        batch_y = batch_y.cpu()
        
        # Verify shape: [batch, height_levels, features]
        if batch_y.dim() != 3:
            raise ValueError(f"Expected 3D tensor [batch, height, features], got shape {batch_y.shape}")
        
        batch_size = batch_y.size(0)
        
        # Process each sample
        for i in range(batch_size):
            # Average over height dimension: [height, features] -> [features]
            sample = batch_y[i].mean(dim=0).double()
            
            count += 1
            
            if mean is None:
                mean = sample.clone()
                if compute_variance:
                    M2 = torch.zeros_like(mean)
            else:
                # Welford's update formulas
                delta = sample - mean
                mean += delta / count
                
                if compute_variance:
                    delta2 = sample - mean
                    M2 += delta * delta2
        
        # Progress logging
        if batch_idx % 100 == 0:
            elapsed = time.time() - start_time
            logger.info(f'Batch {batch_idx}: {count} samples processed ({elapsed:.1f}s)')
    
    if mean is None:
        raise ValueError("No data found in training set")
    
    # Convert to float32
    mean = mean.float()
    
    # Compute final statistics
    elapsed = time.time() - start_time
    logger.info(f'Processed {count} samples in {elapsed:.1f} seconds')
    logger.info(f'Mean shape: {mean.shape}, range: [{mean.min():.6f}, {mean.max():.6f}]')
    
    if compute_variance:
        # Sample variance: divide by (n-1)
        variance = (M2 / (count - 1)).float() if count > 1 else torch.zeros_like(mean)
        logger.info(f'Variance shape: {variance.shape}, range: [{variance.min():.6f}, {variance.max():.6f}]')
        return mean, variance
    else:
        return mean


def main():
    """Main entry point."""
    args = parse_arguments()
    
    logger.info('=' * 50)
    logger.info('Training Set Statistics Computation')
    logger.info('=' * 50)
    logger.info(f'Input dataset: {args.dataset_input}')
    logger.info(f'Output dataset: {args.dataset_output}')
    logger.info(f'Save path: {args.save_path}')
    logger.info(f'Data percentage: {args.percent*100:.1f}%')
    logger.info(f'Subsample rate: {args.subsample*100:.1f}%')
    logger.info(f'Batch size: {args.batch_size}')
    logger.info(f'Workers: {args.num_workers}')
    
    try:
        # Get sorted file lists
        input_files, output_files = get_sorted_file_lists(args.dataset_input, args.dataset_output)
        
        # Select training files (same logic as training script)
        train_input, train_output = select_training_files(input_files, output_files, args.percent)
        
        # Create data loader
        dataset = IconColumnIterableDataset(
            train_input,
            train_output,
            subsample=args.subsample,
            cache_dir='/tmp',
            shuffle=False  # No need to shuffle for statistics
        )
        
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            pin_memory=True,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None
        )
        
        # Compute statistics
        if args.compute_variance:
            mean, variance = compute_statistics_welford(data_loader, compute_variance=True)
            statistics = {
                'train_target_mean': mean,
                'train_target_variance': variance,
                'statistics_type': 'mean_and_variance',
                'n_files': len(train_input),
                'subsample_rate': args.subsample
            }
        else:
            mean = compute_statistics_welford(data_loader, compute_variance=False)
            statistics = {
                'train_target_mean': mean,
                'statistics_type': 'mean_only',
                'n_files': len(train_input),
                'subsample_rate': args.subsample
            }
        
        # Save statistics
        os.makedirs(dirname(args.save_path), exist_ok=True)
        with open(args.save_path, 'wb') as f:
            pickle.dump(statistics, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        file_size = os.path.getsize(args.save_path) / 1024  # KB
        logger.info(f'Saved statistics to: {args.save_path} ({file_size:.1f} KB)')
        
        # Print summary
        print("\n" + "="*50)
        print("COMPUTATION COMPLETE")
        print("="*50)
        print(f"Files processed: {len(train_input)}")
        print(f"Statistics type: {statistics['statistics_type']}")
        print(f"Mean shape: {statistics['train_target_mean'].shape}")
        if 'train_target_variance' in statistics:
            print(f"Variance shape: {statistics['train_target_variance'].shape}")
        print(f"Output saved to: {args.save_path}")
        
    except Exception as e:
        logger.error(f'Error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main() 