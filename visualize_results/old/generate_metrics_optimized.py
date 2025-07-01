import torch
import pickle
import numpy as np
import pandas as pd
from os.path import join
from tabulate import tabulate
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# ====================================================================
# OPTIMIZED METRICS CALCULATION
# ====================================================================

def calculate_metrics_optimized(y_true, y_pred, train_std=None, train_mean=None):
    """Optimized metrics calculation with vectorized operations."""
    
    # Convert to torch tensors if needed (do this once)
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
    
    original_shape = y_true.shape
    
    # Reshape to 2D - combine samples and heights
    if len(original_shape) > 2:
        y_true_2d = y_true.reshape(-1, original_shape[-1])
        y_pred_2d = y_pred.reshape(-1, original_shape[-1])
    else:
        y_true_2d = y_true
        y_pred_2d = y_pred
    
    # Pre-compute common terms to avoid redundant calculations
    diff = y_pred_2d - y_true_2d
    abs_diff = torch.abs(diff)
    squared_diff = diff ** 2
    
    metrics = {}
    
    # Basic metrics (vectorized)
    mae = torch.mean(abs_diff, dim=0)
    mse = torch.mean(squared_diff, dim=0) 
    rmse = torch.sqrt(mse)
    
    metrics['MAE'] = mae
    metrics['RMSE'] = rmse
    metrics['MSE'] = mse
    
    # Normalization factors
    if train_std is not None:
        if isinstance(train_std, np.ndarray):
            train_std = torch.tensor(train_std, dtype=torch.float32)
        
        if len(train_std.shape) > 1:
            train_std_2d = train_std.reshape(-1, train_std.shape[-1])
            sigma = torch.mean(train_std_2d, dim=0)
        else:
            sigma = train_std
    else:
        sigma = torch.std(y_true_2d, dim=0)
    
    # Normalized metrics
    metrics['NRMSE'] = rmse / sigma
    metrics['nMAE'] = mae / sigma
    
    # Test-normalized metrics
    test_std = torch.std(y_true_2d, dim=0)
    metrics['NRMSE_test'] = rmse / test_std
    metrics['nMAE_test'] = mae / test_std
    
    # R-squared (vectorized)
    y_true_mean = torch.mean(y_true_2d, dim=0)
    ss_tot = torch.sum((y_true_2d - y_true_mean.unsqueeze(0)) ** 2, dim=0)
    ss_res = torch.sum(squared_diff, dim=0)
    metrics['R2'] = 1 - (ss_res / ss_tot)
    
    # OPTIMIZED Pearson correlation - fully vectorized
    metrics['Pearson_r'] = calculate_pearson_vectorized(y_true_2d, y_pred_2d)
    
    return metrics

def calculate_pearson_vectorized(y_true_2d, y_pred_2d):
    """Vectorized Pearson correlation calculation - much faster than loop."""
    
    # Center the data (subtract mean)
    y_true_centered = y_true_2d - torch.mean(y_true_2d, dim=0, keepdim=True)
    y_pred_centered = y_pred_2d - torch.mean(y_pred_2d, dim=0, keepdim=True)
    
    # Compute correlation coefficients all at once
    numerator = torch.sum(y_true_centered * y_pred_centered, dim=0)
    
    # Compute denominators
    y_true_ss = torch.sum(y_true_centered ** 2, dim=0)
    y_pred_ss = torch.sum(y_pred_centered ** 2, dim=0) 
    denominator = torch.sqrt(y_true_ss * y_pred_ss)
    
    # Handle potential division by zero
    epsilon = 1e-8
    pearson_r = numerator / (denominator + epsilon)
    
    return pearson_r

def calculate_skill_scores_optimized(model_results, baseline_name, y_true=None, train_mean=None):
    """Optimized skill score calculation."""
    skill_scores = {}
    
    if baseline_name == 'climatology' and y_true is not None and train_mean is not None:
        # Vectorized climatology baseline calculation
        if isinstance(y_true, np.ndarray):
            y_true = torch.tensor(y_true, dtype=torch.float32)
        if isinstance(train_mean, np.ndarray):
            train_mean = torch.tensor(train_mean, dtype=torch.float32)
            
        original_shape = y_true.shape
        if len(original_shape) > 2:
            y_true_2d = y_true.reshape(-1, original_shape[-1])
        else:
            y_true_2d = y_true
            
        if len(train_mean.shape) > 1:
            train_mean_broadcast = train_mean.unsqueeze(0).expand(original_shape[0], -1, -1)
            train_mean_2d = train_mean_broadcast.reshape(-1, original_shape[-1])
        else:
            train_mean_2d = train_mean.unsqueeze(0).expand_as(y_true_2d)
            
        # Vectorized baseline MSE calculation
        baseline_mse = torch.mean((y_true_2d - train_mean_2d) ** 2, dim=0)
        
    else:
        if baseline_name not in model_results:
            return skill_scores
        baseline_mse = model_results[baseline_name]['MSE']
    
    # Vectorized skill score calculation for all models
    for model_name, metrics in model_results.items():
        model_mse = metrics['MSE']
        skill_scores[model_name] = 1 - (model_mse / baseline_mse)
    
    return skill_scores

# ====================================================================
# PARALLEL DATA LOADING
# ====================================================================

def load_model_predictions_parallel(models, max_workers=4):
    """Load predictions for multiple models in parallel."""
    
    def load_single_model(model):
        """Load data for a single model."""
        model_path = model['path']
        model_name = model['name']
        
        print(f'Loading {model_name}...')
        start_time = time.time()
        
        with open(join(model_path, 'y_true.pickle'), 'rb') as handle:
            y_true = pickle.load(handle)
        
        with open(join(model_path, 'y_pred.pickle'), 'rb') as handle:
            y_pred = pickle.load(handle)
        
        # Reshape if needed
        if isinstance(y_true, np.ndarray) and y_true.shape[-1] == 490:
            y_true = y_true.reshape(-1, 70, 7)
            y_pred = y_pred.reshape(-1, 70, 7)
        
        load_time = time.time() - start_time
        print(f'Loaded {model_name} in {load_time:.2f}s - Shape: {y_true.shape}')
        
        return model_name, y_true, y_pred
    
    # Load models in parallel
    model_data = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_model = {executor.submit(load_single_model, model): model for model in models}
        
        for future in as_completed(future_to_model):
            try:
                model_name, y_true, y_pred = future.result()
                model_data[model_name] = (y_true, y_pred)
            except Exception as exc:
                model = future_to_model[future]
                print(f'Model {model["name"]} generated an exception: {exc}')
    
    return model_data

# ====================================================================
# BATCH PROCESSING FOR LARGE DATASETS
# ====================================================================

def calculate_metrics_batched(y_true, y_pred, train_std=None, train_mean=None, batch_size=10000):
    """Calculate metrics in batches to handle large datasets efficiently."""
    
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
    
    original_shape = y_true.shape
    if len(original_shape) > 2:
        y_true_2d = y_true.reshape(-1, original_shape[-1])
        y_pred_2d = y_pred.reshape(-1, original_shape[-1])
    else:
        y_true_2d = y_true
        y_pred_2d = y_pred
    
    n_samples, n_features = y_true_2d.shape
    
    # If dataset is small, use regular calculation
    if n_samples <= batch_size * 2:
        return calculate_metrics_optimized(y_true, y_pred, train_std, train_mean)
    
    print(f"Large dataset detected ({n_samples} samples), using batched calculation...")
    
    # Initialize accumulators
    mae_sum = torch.zeros(n_features)
    mse_sum = torch.zeros(n_features)
    
    # For Pearson correlation - need to accumulate covariances
    y_true_sum = torch.zeros(n_features)
    y_pred_sum = torch.zeros(n_features)
    y_true_sq_sum = torch.zeros(n_features)
    y_pred_sq_sum = torch.zeros(n_features)
    cross_sum = torch.zeros(n_features)
    
    # For R-squared
    ss_res_sum = torch.zeros(n_features)
    
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    # Process in batches
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        
        y_true_batch = y_true_2d[start_idx:end_idx]
        y_pred_batch = y_pred_2d[start_idx:end_idx]
        batch_n = end_idx - start_idx
        
        # Accumulate for MAE and MSE
        mae_sum += torch.sum(torch.abs(y_true_batch - y_pred_batch), dim=0)
        mse_sum += torch.sum((y_true_batch - y_pred_batch) ** 2, dim=0)
        
        # Accumulate for Pearson correlation
        y_true_sum += torch.sum(y_true_batch, dim=0)
        y_pred_sum += torch.sum(y_pred_batch, dim=0)
        y_true_sq_sum += torch.sum(y_true_batch ** 2, dim=0)
        y_pred_sq_sum += torch.sum(y_pred_batch ** 2, dim=0)
        cross_sum += torch.sum(y_true_batch * y_pred_batch, dim=0)
        
        # For R-squared, we need the overall mean
        # Will calculate this after getting the total mean
    
    # Calculate final metrics
    mae = mae_sum / n_samples
    mse = mse_sum / n_samples
    rmse = torch.sqrt(mse)
    
    # Pearson correlation from accumulated statistics
    y_true_mean = y_true_sum / n_samples
    y_pred_mean = y_pred_sum / n_samples
    
    covariance = (cross_sum / n_samples) - (y_true_mean * y_pred_mean)
    y_true_var = (y_true_sq_sum / n_samples) - (y_true_mean ** 2)
    y_pred_var = (y_pred_sq_sum / n_samples) - (y_pred_mean ** 2)
    
    pearson_r = covariance / (torch.sqrt(y_true_var * y_pred_var) + 1e-8)
    
    # R-squared - need second pass for this
    ss_tot = torch.zeros(n_features)
    ss_res = torch.zeros(n_features)
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        
        y_true_batch = y_true_2d[start_idx:end_idx]
        y_pred_batch = y_pred_2d[start_idx:end_idx]
        
        ss_tot += torch.sum((y_true_batch - y_true_mean.unsqueeze(0)) ** 2, dim=0)
        ss_res += torch.sum((y_true_batch - y_pred_batch) ** 2, dim=0)
    
    r2 = 1 - (ss_res / ss_tot)
    
    # Assemble results
    metrics = {
        'MAE': mae,
        'RMSE': rmse, 
        'MSE': mse,
        'Pearson_r': pearson_r,
        'R2': r2
    }
    
    # Add normalized metrics
    if train_std is not None:
        if isinstance(train_std, np.ndarray):
            train_std = torch.tensor(train_std, dtype=torch.float32)
        
        if len(train_std.shape) > 1:
            train_std_2d = train_std.reshape(-1, train_std.shape[-1])
            sigma = torch.mean(train_std_2d, dim=0)
        else:
            sigma = train_std
    else:
        sigma = torch.sqrt(y_true_var)  # Use computed variance
    
    metrics['NRMSE'] = rmse / sigma
    metrics['nMAE'] = mae / sigma
    
    # Test-normalized versions
    test_std = torch.sqrt(y_true_var)
    metrics['NRMSE_test'] = rmse / test_std
    metrics['nMAE_test'] = mae / test_std
    
    return metrics

# ====================================================================
# MEMORY-EFFICIENT PROCESSING
# ====================================================================

def process_models_memory_efficient(models, train_std=None, train_mean=None, 
                                  use_parallel_loading=True, batch_size=10000):
    """Process models with memory optimization."""
    
    print(f"Processing {len(models)} models with optimization...")
    start_time = time.time()
    
    # Load all model data (parallel or sequential)
    if use_parallel_loading and len(models) > 1:
        print("Using parallel data loading...")
        model_data = load_model_predictions_parallel(models, max_workers=min(4, len(models)))
    else:
        print("Using sequential data loading...")
        model_data = {}
        for model in models:
            print(f"Loading {model['name']}...")
            with open(join(model['path'], 'y_true.pickle'), 'rb') as handle:
                y_true = pickle.load(handle)
            with open(join(model['path'], 'y_pred.pickle'), 'rb') as handle:
                y_pred = pickle.load(handle)
            
            if isinstance(y_true, np.ndarray) and y_true.shape[-1] == 490:
                y_true = y_true.reshape(-1, 70, 7)
                y_pred = y_pred.reshape(-1, 70, 7)
            
            model_data[model['name']] = (y_true, y_pred)
    
    loading_time = time.time() - start_time
    print(f"Data loading completed in {loading_time:.2f}s")
    
    # Calculate metrics for each model
    model_results = {}
    calc_start = time.time()
    
    for model_name, (y_true, y_pred) in model_data.items():
        print(f"Calculating metrics for {model_name}...")
        model_start = time.time()
        
        # Choose calculation method based on data size
        total_elements = y_true.size if hasattr(y_true, 'size') else np.prod(y_true.shape)
        
        if total_elements > batch_size * 100:  # Very large dataset
            metrics = calculate_metrics_batched(y_true, y_pred, train_std, train_mean, batch_size)
        else:
            metrics = calculate_metrics_optimized(y_true, y_pred, train_std, train_mean)
        
        model_results[model_name] = metrics
        
        model_time = time.time() - model_start
        print(f"  {model_name} completed in {model_time:.2f}s")
    
    calc_time = time.time() - calc_start
    total_time = time.time() - start_time
    
    print(f"\nOptimization Summary:")
    print(f"  Data loading: {loading_time:.2f}s")
    print(f"  Metrics calculation: {calc_time:.2f}s") 
    print(f"  Total time: {total_time:.2f}s")
    
    return model_results

# ====================================================================
# USAGE EXAMPLE
# ====================================================================

def main_optimized():
    """Main function using optimized metrics calculation."""
    
    # Your model configuration (same as before)
    models = [
        {'name': 'ViT-1D-128-l4', 'path': '/mydata/deepcloud/yves/results-new/vit_1d_128_l4_triangle/test'},
        {'name': 'GT-2D-1024-L2', 'path': '/mydata/deepcloud/yves/results-new/gt_gencast_1024_l2_triangle39_k2_new/test'},
        {'name': 'GT-3D-128-L4', 'path': '/mydata/deepcloud/yves/results-new/gt_simplified_128_l4/test'},
    ]
    
    # Load training statistics
    TRAINING_STATS_PATH = '/mydata/deepcloud/yves/h5_tendency_data_all/train_target_statistics.pickle'
    
    try:
        with open(TRAINING_STATS_PATH, 'rb') as f:
            stats_dict = pickle.load(f)
        train_std = np.sqrt(stats_dict['train_target_variance'])
        train_mean = stats_dict.get('train_target_mean', None)
        print(f"Loaded training statistics: std shape {train_std.shape}")
    except:
        print("Using test set normalization")
        train_std, train_mean = None, None
    
    # Process models with optimization
    model_results = process_models_memory_efficient(
        models, 
        train_std=train_std, 
        train_mean=train_mean,
        use_parallel_loading=True,
        batch_size=50000  # Adjust based on your memory
    )
    
    # Calculate skill scores (optimized)
    baseline_name = 'climatology'
    y_true_for_skill = None
    
    # Get y_true for skill score calculation
    if baseline_name == 'climatology':
        # Load any model's y_true (they should be the same)
        first_model = models[0]
        with open(join(first_model['path'], 'y_true.pickle'), 'rb') as handle:
            y_true_for_skill = pickle.load(handle)
        if isinstance(y_true_for_skill, np.ndarray) and y_true_for_skill.shape[-1] == 490:
            y_true_for_skill = y_true_for_skill.reshape(-1, 70, 7)
    
    skill_scores = calculate_skill_scores_optimized(
        model_results, baseline_name, y_true_for_skill, train_mean
    )
    
    # Add skill scores to results
    for model_name in model_results:
        if model_name in skill_scores:
            model_results[model_name]['Skill_Score'] = skill_scores[model_name]
    
    print("\n=== OPTIMIZED RESULTS ===")
    for model_name, metrics in model_results.items():
        print(f"\n{model_name}:")
        if 'NRMSE' in metrics:
            nrmse_mean = metrics['NRMSE'].mean().item()
            print(f"  Average nRMSE: {nrmse_mean:.4f}")
        if 'Pearson_r' in metrics:
            pearson_mean = metrics['Pearson_r'].mean().item()
            print(f"  Average Pearson r: {pearson_mean:.4f}")
        if 'Skill_Score' in metrics:
            skill_mean = metrics['Skill_Score'].mean().item()
            print(f"  Average Skill Score: {skill_mean:.4f}")

if __name__ == "__main__":
    main_optimized() 