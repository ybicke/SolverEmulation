"""
Drop-in replacement for memory-hungry precompute_train_target_mean function.
This version computes statistics incrementally without loading all data into memory.
"""

import torch
import logging

logger = logging.getLogger(__name__)


def precompute_train_target_mean_memory_efficient(train_set):
    """
    Memory-efficient version of precompute_train_target_mean.
    Computes mean incrementally using Welford's online algorithm.
    
    This is a drop-in replacement for the original function that was causing OOM errors.
    """
    logger.info('Computing target statistics memory-efficiently from training data...')
    
    # Initialize for incremental computation
    count = 0
    mean = None
    
    for batch_data in train_set:
        # Extract target data (assumes format: _, _, batch_y, _)
        if len(batch_data) == 4:
            _, _, batch_y, _ = batch_data
        else:
            batch_y = batch_data[2] if len(batch_data) > 2 else batch_data[1]
        
        # Move to CPU to save memory
        batch_y = batch_y.cpu()
        
        # Handle different tensor shapes
        if batch_y.dim() == 4:
            # Shape: [Batch, Columns, Height, Features] -> average over columns
            batch_y = batch_y.mean(dim=1)  # Now [Batch, Height, Features]
        
        # Process each sample in the batch
        batch_size = batch_y.size(0)
        
        for i in range(batch_size):
            if batch_y.dim() == 3:
                # For 3D: [Height, Features] - average over height dimension
                sample = batch_y[i].mean(dim=0)  # [Features]
            else:
                # For 2D: [Features] - use directly
                sample = batch_y[i]
            
            count += 1
            
            if mean is None:
                mean = sample.clone().double()
            else:
                # Welford's online algorithm for mean
                delta = sample.double() - mean
                mean += delta / count
    
    if mean is None:
        raise ValueError("No data found in training set")
    
    # Convert back to float32 and ensure proper shape
    train_target_mean = mean.float()
    
    logger.info(f'Computed mean from {count} samples')
    logger.info(f'Mean shape: {train_target_mean.shape}')
    logger.info(f'Mean range: [{train_target_mean.min():.8f}, {train_target_mean.max():.8f}]')
    
    return train_target_mean


def precompute_train_target_mean_and_var_memory_efficient(train_set):
    """
    Memory-efficient computation of both mean and variance.
    Returns both for advanced evaluation metrics.
    """
    logger.info('Computing target mean and variance memory-efficiently from training data...')
    
    # Initialize for incremental computation
    count = 0
    mean = None
    M2 = None  # Sum of squares of differences from mean
    
    for batch_data in train_set:
        # Extract target data
        if len(batch_data) == 4:
            _, _, batch_y, _ = batch_data
        else:
            batch_y = batch_data[2] if len(batch_data) > 2 else batch_data[1]
        
        # Move to CPU
        batch_y = batch_y.cpu()
        
        # Handle different tensor shapes
        if batch_y.dim() == 4:
            batch_y = batch_y.mean(dim=1)  # Average over columns
        
        # Process each sample
        batch_size = batch_y.size(0)
        
        for i in range(batch_size):
            if batch_y.dim() == 3:
                sample = batch_y[i].mean(dim=0)  # Average over height
            else:
                sample = batch_y[i]
            
            count += 1
            
            if mean is None:
                mean = sample.clone().double()
                M2 = torch.zeros_like(mean)
            else:
                # Welford's online algorithm
                delta = sample.double() - mean
                mean += delta / count
                delta2 = sample.double() - mean
                M2 += delta * delta2
    
    if mean is None:
        raise ValueError("No data found in training set")
    
    # Compute variance
    if count < 2:
        variance = torch.zeros_like(mean)
    else:
        variance = M2 / (count - 1)  # Sample variance
    
    # Convert back to float32
    train_target_mean = mean.float()
    train_target_var = variance.float()
    
    logger.info(f'Computed mean and variance from {count} samples')
    logger.info(f'Mean shape: {train_target_mean.shape}')
    logger.info(f'Variance shape: {train_target_var.shape}')
    
    return train_target_mean, train_target_var


# Legacy compatibility function
def precompute_train_target_mean(train_set):
    """
    Legacy compatibility wrapper - calls the memory-efficient version.
    This is a direct replacement for the original OOM-causing function.
    """
    return precompute_train_target_mean_memory_efficient(train_set) 