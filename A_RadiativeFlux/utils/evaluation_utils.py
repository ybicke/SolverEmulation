#!/usr/bin/env python3
"""
Evaluation utilities for model performance analysis including timing statistics
and test result processing.
"""


import torch
from os.path import join


def process_timing_statistics(timing_data, model, test_path, args, count_parameters_func, test_loss=None, test_mae=None):
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
Hardware: {gpu_info}
Parameters: {count_parameters_func(model):,}

Configuration:
Hidden Dimension: {args.hidden_dim}
Layers: {args.layers}
Batch Size: {batch_size}
"""
    
    # Add test performance if provided
    if test_loss is not None or test_mae is not None:
        summary_text += "\nTest Performance:\n"
        if test_loss is not None:
            summary_text += f"Test Loss: {test_loss:.6f}\n"
        if test_mae is not None:
            summary_text += f"Test MAE: {test_mae:.6f}\n"
    
    summary_text += f"""
Batch Performance:
Batches measured: {len(timing_data)}
Mean batch time: {mean_time*1000:.3f} ± {std_time*1000:.3f} ms/batch
Min/Max batch time: {min_time*1000:.3f}/{max_time*1000:.3f} ms/batch

Sample Performance:
Per sample time: {per_sample_time*1000:.3f} ms/sample
Throughput: {samples_per_second:.1f} samples/second
"""
    
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


def warm_up_model(model, test_loader, normalizer, device, num_warmup_batches=10):
    """
    Perform model warmup for accurate timing measurements.
    
    Args:
        model: The model to warm up
        test_loader: Test data loader
        normalizer: Data normalizer
        device: Compute device
        num_warmup_batches: Number of warmup batches
    """
    print(f'Performing {num_warmup_batches} warmup passes...')
    warmup_iter = iter(test_loader)
    for _ in range(num_warmup_batches):
        try:
            data = next(warmup_iter)
            batch_x3, batch_x2, _ = data
            batch_x3, batch_x2 = batch_x3.to(device), batch_x2.to(device)
            batch_x3_norm, batch_x2_norm, batch_x2_orig = normalizer.normalize(batch_x3, batch_x2)
            with torch.no_grad():
                _ = model(batch_x3_norm, batch_x2_norm, batch_x2_orig)
        except StopIteration:
            break
    
    if torch.cuda.is_available():
        torch.cuda.synchronize() 