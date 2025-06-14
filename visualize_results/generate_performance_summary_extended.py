import torch
import pickle
import numpy as np
import pandas as pd
from os.path import join
from tabulate import tabulate
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ====================================================================
# CONFIGURATION SECTION
# ====================================================================

# Models to compare
models = [
    {'name': 'GNN-3D-64-L2-100', 'path': '/mydata/deepcloud/yves/results-temp/gnn3d_id39_tendency_64_l2_100/test'},
    {'name': 'GNN-1D-64-L2-100', 'path': '/mydata/deepcloud/yves/results-temp/gnn_64_l2_tendency_1d_triangle_fully_connected/test'},
]

# Paths
TRAINING_STATS_PATH = '/mydata/deepcloud/yves/h5_tendency_data_all/train_target_statistics.pickle'
PRESSURE_WEIGHTS_PATH = None  # Set to path of pressure weights if available
OUTPUT_DIR = '/mydata/deepcloud/yves/final_results'
OUTPUT_FILENAME = '3D_vs_1D_tendency_eval_extended.md'

# Define target names and units
target_units = {
    "Sum of Temperature Tendency": "K s-1", 
    "Dynamical Temperature Tendency": "K s-1",
    "Sum of Zonal Wind Tendency": "m s-2",
    "Sum of Meridional Wind Tendency": "m s-2",
    "Convective Tend. Absolute Humidity": "kg m-3 s-1",
    "Convective Tend. Cloud Water Mass Density": "kg m-3 s-1",
    "Convective Tend. Cloud Ice Mass Density": "kg m-3 s-1"
}

# ====================================================================
# EXTENDED METRICS CALCULATION
# ====================================================================

def calculate_extended_metrics(y_true, y_pred, train_std=None, pressure_weights=None):
    """Calculate comprehensive metrics including distributional and physical consistency checks."""
    
    # Convert to torch tensors
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
    
    # Get shapes
    original_shape = y_true.shape  # [batch, height, features]
    n_samples, n_levels, n_features = original_shape
    
    # Initialize results dictionary
    metrics = {}
    
    # === 1. STANDARD METRICS (as before) ===
    # Reshape for computation
    y_true_2d = y_true.reshape(-1, n_features)
    y_pred_2d = y_pred.reshape(-1, n_features)
    
    # Basic metrics
    mae = torch.mean(torch.abs(y_true_2d - y_pred_2d), dim=0)
    mse = torch.mean((y_true_2d - y_pred_2d) ** 2, dim=0)
    rmse = torch.sqrt(mse)
    
    metrics['MAE'] = mae
    metrics['RMSE'] = rmse
    
    # NRMSE
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
    
    epsilon = 1e-8
    nrmse = rmse / (sigma + epsilon)
    metrics['NRMSE'] = nrmse
    
    # R-squared
    y_true_mean = torch.mean(y_true_2d, dim=0)
    ss_tot = torch.sum((y_true_2d - y_true_mean.unsqueeze(0)) ** 2, dim=0)
    ss_res = torch.sum((y_true_2d - y_pred_2d) ** 2, dim=0)
    r2 = 1 - (ss_res / (ss_tot + epsilon))
    metrics['R2'] = r2
    
    # === 2. BIAS METRICS ===
    # Mean bias (systematic error)
    bias = torch.mean(y_pred_2d - y_true_2d, dim=0)
    metrics['Bias'] = bias
    
    # Relative bias (as percentage of mean absolute value)
    rel_bias = bias / (torch.mean(torch.abs(y_true_2d), dim=0) + epsilon) * 100
    metrics['RelBias%'] = rel_bias
    
    # === 3. DISTRIBUTIONAL METRICS ===
    # Compute per feature
    percentile_errors = torch.zeros((3, n_features))  # 25th, 50th, 75th percentiles
    iqr_ratio = torch.zeros(n_features)
    
    for feat in range(n_features):
        errors = torch.abs(y_pred_2d[:, feat] - y_true_2d[:, feat])
        percentile_errors[0, feat] = torch.quantile(errors, 0.25)
        percentile_errors[1, feat] = torch.quantile(errors, 0.50)  # Median absolute error
        percentile_errors[2, feat] = torch.quantile(errors, 0.75)
        
        # IQR ratio: (75th - 25th) / 50th - measures error spread
        iqr = percentile_errors[2, feat] - percentile_errors[0, feat]
        iqr_ratio[feat] = iqr / (percentile_errors[1, feat] + epsilon)
    
    metrics['MAE_25th'] = percentile_errors[0]
    metrics['MAE_median'] = percentile_errors[1]
    metrics['MAE_75th'] = percentile_errors[2]
    metrics['IQR_ratio'] = iqr_ratio
    
    # === 4. EXTREME VALUE METRICS ===
    # 95th and 99th percentile errors (for outlier analysis)
    p95_errors = torch.zeros(n_features)
    p99_errors = torch.zeros(n_features)
    
    for feat in range(n_features):
        errors = torch.abs(y_pred_2d[:, feat] - y_true_2d[:, feat])
        p95_errors[feat] = torch.quantile(errors, 0.95)
        p99_errors[feat] = torch.quantile(errors, 0.99)
    
    metrics['MAE_95th'] = p95_errors
    metrics['MAE_99th'] = p99_errors
    
    # === 5. VERTICAL PROFILE METRICS ===
    # Per-level RMSE to see vertical structure of errors
    level_rmse = torch.zeros((n_levels, n_features))
    for level in range(n_levels):
        level_errors = (y_pred[:, level, :] - y_true[:, level, :]) ** 2
        level_rmse[level] = torch.sqrt(torch.mean(level_errors, dim=0))
    
    metrics['Level_RMSE'] = level_rmse
    
    # === 6. PRESSURE-WEIGHTED METRICS (if weights provided) ===
    if pressure_weights is not None:
        if isinstance(pressure_weights, np.ndarray):
            pressure_weights = torch.tensor(pressure_weights, dtype=torch.float32)
        
        # Normalize weights
        weights = pressure_weights / torch.sum(pressure_weights)
        
        # Weighted MAE and RMSE
        weighted_mae = torch.zeros(n_features)
        weighted_rmse = torch.zeros(n_features)
        
        for feat in range(n_features):
            # Compute per-level errors
            level_mae = torch.mean(torch.abs(y_pred[:, :, feat] - y_true[:, :, feat]), dim=0)
            level_mse = torch.mean((y_pred[:, :, feat] - y_true[:, :, feat]) ** 2, dim=0)
            
            # Apply pressure weighting
            weighted_mae[feat] = torch.sum(weights * level_mae)
            weighted_rmse[feat] = torch.sqrt(torch.sum(weights * level_mse))
        
        metrics['MAE_weighted'] = weighted_mae
        metrics['RMSE_weighted'] = weighted_rmse
    
    # === 7. PHYSICAL CONSISTENCY CHECKS ===
    # Energy conservation check (for temperature tendency)
    # Sum of heating rates should be physically reasonable
    temp_tend_sum = torch.sum(y_pred[:, :, 0], dim=1)  # Sum over height
    temp_tend_sum_true = torch.sum(y_true[:, :, 0], dim=1)
    energy_conservation_error = torch.std(temp_tend_sum - temp_tend_sum_true)
    metrics['Energy_Conservation_Error'] = energy_conservation_error
    
    # === 8. CORRELATION STRUCTURE ===
    # Spatial correlation of errors (do errors cluster?)
    error_correlation = torch.zeros((n_features, n_features))
    errors_all = y_pred_2d - y_true_2d
    
    for i in range(n_features):
        for j in range(n_features):
            if i <= j:  # Only compute upper triangle
                corr = torch.corrcoef(torch.stack([errors_all[:, i], errors_all[:, j]]))[0, 1]
                error_correlation[i, j] = corr
                error_correlation[j, i] = corr
    
    metrics['Error_Correlation'] = error_correlation
    
    return metrics

# ====================================================================
# LOAD AND PROCESS DATA
# ====================================================================

def load_training_statistics(stats_path):
    """Load pre-computed training statistics."""
    try:
        with open(stats_path, 'rb') as f:
            stats_dict = pickle.load(f)
        
        if 'train_target_variance' in stats_dict:
            train_variance = stats_dict['train_target_variance']
            train_std = np.sqrt(train_variance)
            return train_std
        else:
            return None
    except:
        return None

def load_pressure_weights(weights_path):
    """Load pressure level weights if available."""
    if weights_path is None:
        return None
    try:
        weights = np.load(weights_path)
        return weights
    except:
        print(f"Could not load pressure weights from {weights_path}")
        return None

# ====================================================================
# FORMATTING AND OUTPUT
# ====================================================================

def create_extended_summary(model_results, target_names):
    """Create comprehensive summary with all metrics."""
    
    output_text = """# Extended Model Performance Analysis

This comprehensive analysis includes standard metrics plus distributional statistics, 
vertical profiles, and physical consistency checks.

## 1. Standard Metrics Summary

"""
    
    # Standard metrics table (as before)
    standard_metrics = ['MAE', 'RMSE', 'NRMSE', 'R2', 'Bias']
    for metric in standard_metrics:
        output_text += f"\n### {metric}\n\n"
        
        # Create DataFrame
        data = {}
        for model_name, metrics in model_results.items():
            if metric in metrics:
                data[model_name] = metrics[metric].numpy()
        
        df = pd.DataFrame(data, index=target_names)
        
        # Format based on metric type
        if metric == 'R2':
            output_text += "Higher values are better.\n\n"
            formatted_df = df.map(lambda x: f"{x:.4f}")
        elif metric == 'Bias':
            output_text += "Values closer to zero are better.\n\n"
            formatted_df = df.map(lambda x: f"{x:+.4e}")  # Show sign
        else:
            output_text += "Lower values are better.\n\n"
            formatted_df = df.map(lambda x: f"{x:.4e}")
        
        output_text += formatted_df.to_markdown() + "\n"
    
    # Distributional analysis
    output_text += "\n## 2. Error Distribution Analysis\n\n"
    output_text += "### Percentile Error Analysis\n\n"
    output_text += "Shows 25th, 50th (median), and 75th percentile of absolute errors.\n\n"
    
    for model_name in model_results.keys():
        output_text += f"\n**{model_name}:**\n\n"
        
        percentile_data = {
            '25th percentile': model_results[model_name]['MAE_25th'].numpy(),
            'Median (50th)': model_results[model_name]['MAE_median'].numpy(),
            '75th percentile': model_results[model_name]['MAE_75th'].numpy(),
            'IQR Ratio': model_results[model_name]['IQR_ratio'].numpy()
        }
        
        df = pd.DataFrame(percentile_data, index=target_names)
        formatted_df = df.map(lambda x: f"{x:.4e}")
        output_text += formatted_df.to_markdown() + "\n"
    
    # Extreme value analysis
    output_text += "\n### Extreme Error Analysis\n\n"
    output_text += "95th and 99th percentile errors show worst-case performance.\n\n"
    
    for model_name in model_results.keys():
        output_text += f"\n**{model_name}:**\n\n"
        
        extreme_data = {
            '95th percentile': model_results[model_name]['MAE_95th'].numpy(),
            '99th percentile': model_results[model_name]['MAE_99th'].numpy(),
            'Ratio 99th/median': (model_results[model_name]['MAE_99th'] / 
                                 (model_results[model_name]['MAE_median'] + 1e-8)).numpy()
        }
        
        df = pd.DataFrame(extreme_data, index=target_names)
        formatted_df = df.map(lambda x: f"{x:.4e}")
        output_text += formatted_df.to_markdown() + "\n"
    
    # Vertical profile summary
    output_text += "\n## 3. Vertical Error Structure\n\n"
    output_text += "Average RMSE by pressure level (top to bottom of atmosphere).\n\n"
    
    for model_name in model_results.keys():
        output_text += f"\n**{model_name}:**\n\n"
        level_rmse = model_results[model_name]['Level_RMSE'].numpy()
        
        # Show a few key levels
        n_levels = level_rmse.shape[0]
        key_levels = [0, n_levels//4, n_levels//2, 3*n_levels//4, n_levels-1]
        level_names = ['Top', 'Upper-mid', 'Mid', 'Lower-mid', 'Bottom']
        
        level_data = {}
        for i, (level_idx, level_name) in enumerate(zip(key_levels, level_names)):
            level_data[f'Level {level_name}'] = level_rmse[level_idx]
        
        df = pd.DataFrame(level_data, index=target_names)
        formatted_df = df.map(lambda x: f"{x:.4e}")
        output_text += formatted_df.to_markdown() + "\n"
    
    # Physical consistency
    output_text += "\n## 4. Physical Consistency Checks\n\n"
    
    for model_name in model_results.keys():
        energy_error = model_results[model_name].get('Energy_Conservation_Error', None)
        if energy_error is not None:
            output_text += f"\n**{model_name}:**\n"
            output_text += f"- Energy conservation error (temp tendency): {energy_error:.4e}\n"
    
    # Error correlation
    output_text += "\n## 5. Error Correlation Analysis\n\n"
    output_text += "How correlated are the errors between different variables?\n"
    output_text += "High correlation suggests systematic model biases.\n\n"
    
    for model_name in model_results.keys():
        output_text += f"\n**{model_name}:**\n\n"
        error_corr = model_results[model_name]['Error_Correlation'].numpy()
        
        # Show only significant correlations
        output_text += "Significant error correlations (|r| > 0.5):\n"
        for i in range(len(target_names)):
            for j in range(i+1, len(target_names)):
                if abs(error_corr[i, j]) > 0.5:
                    output_text += f"- {target_names[i]} vs {target_names[j]}: {error_corr[i, j]:.3f}\n"
    
    return output_text

# ====================================================================
# MAIN EXECUTION
# ====================================================================

def main():
    print("Starting extended model performance evaluation...")
    
    # Load data
    train_std = load_training_statistics(TRAINING_STATS_PATH)
    pressure_weights = load_pressure_weights(PRESSURE_WEIGHTS_PATH)
    
    # Store results
    model_results = {}
    
    # Process each model
    for model in models:
        print(f"\nProcessing {model['name']}...")
        
        # Load predictions
        with open(join(model['path'], 'y_true.pickle'), 'rb') as f:
            y_true = pickle.load(f)
        with open(join(model['path'], 'y_pred.pickle'), 'rb') as f:
            y_pred = pickle.load(f)
        
        # Reshape if needed
        if isinstance(y_true, np.ndarray) and y_true.shape[-1] == 490:
            y_true = y_true.reshape(-1, 70, 7)
            y_pred = y_pred.reshape(-1, 70, 7)
        
        # Calculate extended metrics
        metrics = calculate_extended_metrics(y_true, y_pred, train_std, pressure_weights)
        model_results[model['name']] = metrics
    
    # Create summary
    target_names = list(target_units.keys())
    output_text = create_extended_summary(model_results, target_names)
    
    # Save results
    output_path = join(OUTPUT_DIR, OUTPUT_FILENAME)
    with open(output_path, 'w') as f:
        f.write(output_text)
    
    print(f"\nExtended analysis complete!")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main() 