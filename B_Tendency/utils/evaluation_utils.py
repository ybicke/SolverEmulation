#!/usr/bin/env python3
"""
Evaluation utilities for tendency model performance analysis including timing statistics
and test result processing.
"""

import torch
from os.path import join


def process_timing_statistics(timing_data, model, test_path, args, count_parameters_func, test_loss=None, test_mae=None, num_warmup_batches=None, total_test_batches=None):
    """
    Process and report timing statistics from collected data.
    
    Args:
        timing_data: List of timing measurements in seconds
        model: The trained model
        test_path: Path to save timing results
        args: Training arguments
        count_parameters_func: Function to count model parameters
        test_loss: Final test loss value (optional)
        test_mae: Final test MAE value (optional)
        num_warmup_batches: Number of warmup batches used (optional)
        total_test_batches: Total number of test batches processed (optional)
    """
    batch_size = args.batch_size
    
    # Calculate basic statistics
    mean_time = sum(timing_data) / len(timing_data)
    std_time = (sum((t - mean_time) ** 2 for t in timing_data) / len(timing_data)) ** 0.5
    min_time = min(timing_data)
    max_time = max(timing_data)
    
    # Calculate per-sample metrics
    per_sample_time = mean_time / batch_size
    samples_per_second = batch_size / mean_time
    
    gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    
    summary_text = f"""
==================== INFERENCE TIMING SUMMARY ====================
Model: {args.model}
Mode: {args.mode}
Dataset Type: {args.dataset_type}
Hardware: {gpu_info}
Parameters: {count_parameters_func(model):,}

Configuration:
Hidden Dimension: {args.hidden_dim}
Layers: {args.layers}
Batch Size: {batch_size}
Triangle ID: {args.triangle_id}
Division Factor: {args.triangle_division_factor}
"""
    
    # Add model-specific parameters
    if hasattr(args, 'heads') and args.heads:
        summary_text += f"Attention Heads: {args.heads}\n"
    if hasattr(args, 'dim_head') and args.dim_head:
        summary_text += f"Dimension per Head: {args.dim_head}\n"
    if hasattr(args, 'mlp_ratio') and args.mlp_ratio:
        summary_text += f"MLP Ratio: {args.mlp_ratio}\n"
    if hasattr(args, 'dropout') and args.dropout:
        summary_text += f"Dropout: {args.dropout}\n"
    if hasattr(args, 'max_hops') and args.max_hops:
        summary_text += f"Max Hops: {args.max_hops}\n"
    if hasattr(args, 'fully_connected') and args.fully_connected is not None:
        summary_text += f"Fully Connected: {args.fully_connected}\n"
    
    # Add test performance if provided
    if test_loss is not None or test_mae is not None:
        summary_text += "\nTest Performance:\n"
        if test_loss is not None:
            summary_text += f"Test Loss (MSE): {test_loss:.6f}\n"
        if test_mae is not None:
            summary_text += f"Test MAE: {test_mae:.6f}\n"
    
    summary_text += f"""
Evaluation Setup:
"""
    if num_warmup_batches is not None:
        summary_text += f"Warmup batches: {num_warmup_batches}\n"
    if total_test_batches is not None:
        summary_text += f"Total test batches: {total_test_batches}\n"
    
    summary_text += f"""Timing batches measured: {len(timing_data)}

Batch Performance:
Mean batch time: {mean_time*1000:.3f} ± {std_time*1000:.3f} ms/batch
Min/Max batch time: {min_time*1000:.3f}/{max_time*1000:.3f} ms/batch

Sample Performance:
Per sample time: {per_sample_time*1000:.3f} ms/sample
Throughput: {samples_per_second:.1f} samples/second
"""
    
    # Add mode-specific information
    if args.mode == '1d':
        summary_text += f"\n1D Mode (Column-wise):\n"
        summary_text += f"Columns per batch: {batch_size}\n"
        summary_text += f"Time per column: {per_sample_time*1000:.3f} ms\n"
    elif args.mode == '3d':
        summary_text += f"\n3D Mode (Triangle-wise):\n"
        summary_text += f"Triangles per batch: {batch_size}\n"
        summary_text += f"Time per triangle: {per_sample_time*1000:.3f} ms\n"
        if args.dataset_type == 'triangle':
            triangle_size = 1024 // (args.triangle_division_factor ** 2)
            summary_text += f"Columns per triangle: {triangle_size}\n"
            summary_text += f"Time per column: {per_sample_time*1000/triangle_size:.3f} ms\n"
    
    summary_text += "\nTiming Distribution (percentiles):\n"
    sorted_times = sorted(timing_data)
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        idx = int(len(sorted_times) * p / 100)
        summary_text += f"  {p}th percentile: {sorted_times[idx]*1000:.3f} ms\n"
        
    summary_text += "================================================================\n"
    
    # Print to console
    print(summary_text)
    
    # Save human-readable summary to text file
    with open(join(test_path, 'inference_timing_summary.txt'), 'w') as f:
        f.write(summary_text)
    
    return {
        'mean_time': mean_time,
        'std_time': std_time,
        'per_sample_time': per_sample_time,
        'samples_per_second': samples_per_second,
    }


def warm_up_model(model, test_loader, normalizer, target_means, target_vars, device, mode, num_warmup_batches=10):
    """
    Perform model warmup for accurate timing measurements.
    
    Args:
        model: The model to warm up
        test_loader: Test data loader
        normalizer: Data normalizer
        target_means: Target normalization means
        target_vars: Target normalization variances
        device: Compute device
        mode: Training mode ('1d' or '3d')
        num_warmup_batches: Number of warmup batches
        
    Returns:
        int: Actual number of warmup batches performed
    """
    from utils.data_utils import interpolate_w_to_full_levels, transform_targets
    
    print(f'Performing {num_warmup_batches} warmup passes...')
    warmup_iter = iter(test_loader)
    actual_warmup_batches = 0
    
    for i in range(num_warmup_batches):
        try:
            data = next(warmup_iter)
            batch_x3, batch_x2, batch_y, batch_w = data
            batch_x3, batch_x2, batch_y, batch_w = (
                batch_x3.to(device), batch_x2.to(device), 
                batch_y.to(device), batch_w.to(device)
            )
            
            # Process data same as in testing
            w_full = interpolate_w_to_full_levels(batch_w, mode=mode)
            batch_x3_with_w = torch.cat([batch_x3, w_full], dim=-1)
            batch_x3_norm, batch_x2_norm, _ = normalizer.normalize(batch_x3_with_w, batch_x2)
            
            with torch.no_grad():
                _ = model(batch_x3_norm, batch_x2_norm)
            
            actual_warmup_batches += 1
            
        except StopIteration:
            break
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    print(f'Completed {actual_warmup_batches} warmup batches')
    return actual_warmup_batches


def create_test_summary(test_path, args, model, count_parameters_func, test_loss, test_mae, test_time, total_samples):
    """
    Create a comprehensive test summary with all important metrics.
    
    Args:
        test_path: Path to save summary
        args: Training arguments
        model: The trained model
        count_parameters_func: Function to count model parameters
        test_loss: Final test loss
        test_mae: Final test MAE
        test_time: Total test time in seconds
        total_samples: Total number of test samples
    """
    gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    
    summary_text = f"""
==================== TEST SUMMARY ====================
Model: {args.model}
Mode: {args.mode}
Dataset Type: {args.dataset_type}
Hardware: {gpu_info}
Parameters: {count_parameters_func(model):,}

Data Configuration:
Triangle ID: {args.triangle_id}
Division Factor: {args.triangle_division_factor}
Input Channels 3D: {args.channel_3d}
Input Channels 2D: {args.channel_2d}
Output Channels: {args.channels_out}
Height Levels: {args.height}

Model Configuration:
Hidden Dimension: {args.hidden_dim}
Layers: {args.layers}
Batch Size: {args.batch_size}
"""
    
    # Add model-specific parameters
    if hasattr(args, 'heads') and args.heads:
        summary_text += f"Attention Heads: {args.heads}\n"
    if hasattr(args, 'dim_head') and args.dim_head:
        summary_text += f"Dimension per Head: {args.dim_head}\n"
    if hasattr(args, 'mlp_ratio') and args.mlp_ratio:
        summary_text += f"MLP Ratio: {args.mlp_ratio}\n"
    if hasattr(args, 'dropout') and args.dropout:
        summary_text += f"Dropout: {args.dropout}\n"
    if hasattr(args, 'max_hops') and args.max_hops:
        summary_text += f"Max Hops: {args.max_hops}\n"
    if hasattr(args, 'fully_connected') and args.fully_connected is not None:
        summary_text += f"Fully Connected: {args.fully_connected}\n"
    
    summary_text += f"""
Test Results:
Total Samples: {total_samples:,}
Test Loss (MSE): {test_loss:.6f}
Test MAE: {test_mae:.6f}
Total Test Time: {test_time:.2f} seconds
Average Time per Sample: {test_time/total_samples*1000:.3f} ms
Throughput: {total_samples/test_time:.1f} samples/second

Training Configuration:
Learning Rate: {args.learning_rate}
Optimizer: {args.optimizer}
Epochs: {args.num_epoch}
Gradient Clipping: {args.clip}
"""
    
    if args.mode == '1d':
        summary_text += f"\n1D Mode Details:\n"
        summary_text += f"Processing: Column-wise\n"
        summary_text += f"Samples per file: ~1024 columns\n"
    elif args.mode == '3d':
        summary_text += f"\n3D Mode Details:\n"
        summary_text += f"Processing: Triangle-wise\n"
        triangle_size = 1024 // (args.triangle_division_factor ** 2)
        summary_text += f"Columns per triangle: {triangle_size}\n"
    
    summary_text += "================================================\n"
    
    # Print to console
    print(summary_text)
    
    # Save to file
    with open(join(test_path, 'test_summary.txt'), 'w') as f:
        f.write(summary_text)
    
    return summary_text 