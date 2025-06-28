import torch
import pickle
import numpy as np
import pandas as pd
from os.path import join
from tabulate import tabulate

# ====================================================================
# CONFIGURATION SECTION - Modify paths and settings here
# ====================================================================

# Models to compare
models = [
    
    
    #{'name': 'GNN-128-l4', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results/gnn_128_l4/test'},
    #{'name': 'ViT-128-l4', 'path': '/mydata/deepcloud/yves/results_git/vit_column_128_l4_h6_concat/test'},
    #{'name': 'AFNO-128-l4', 'path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_clean/test'},
    #{'name': 'BiLSTM-medium', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results/rnn_medium/test'},
    
    
    #{'name': 'GNN-3D-64-L2-100', 'path': '/mydata/deepcloud/yves/results-temp/gnn3d_id39_tendency_64_l2_100_new/test'},
    #{'name': 'GNN-1D-64-L2-100', 'path': '/mydata/deepcloud/yves/results-temp/gnn_64_l2_tendency_1d_triangle_fully_connected/test'},
    
    
    #{'name': 'GNN-3D-64-L2-100', 'path': '/mydata/deepcloud/yves/results-new/gnn_3d_tendency_64_l2_100_k4_correct_variance/test'},
    #{'name': 'GT-3D-64-L4-k2-100', 'path': '/mydata/deepcloud/yves/results-new/gt_simplified_64_l4_drop03_triangle39_k2/test'},



   # lonlat 1D 3D 128
    {'name': 'GNN-1D-128-L2', 'path': '/mydata/deepcloud/yves/results-new/gnn_1d_tendency_128_l2_100/test'},
    
    # 2D graphcast style
    {'name': 'GNN-2D-1024-L2', 'path': '/mydata/deepcloud/yves/results-new/gnn_2d_graphcast_style_1024_l2_100/test'},
    
    {'name': 'GNN-3D-128-L2-lonlat', 'path': '/mydata/deepcloud/yves/results-new/gnn_3d_tendency_128_l2_100_lonlat/test'},   

]

# Path to training statistics (for NRMSE normalization)
# This pickle file should contain a dictionary with at least 'train_target_variance' key
# The variance will be converted to std for NRMSE normalization
# Ideally computed from 100% of training data for robust estimates
TRAINING_STATS_PATH = '/mydata/deepcloud/yves/h5_tendency_data_all/train_target_statistics.pickle'

# Path to pressure level information (optional, for mass-weighted metrics)
# Should be a numpy array of pressure differences (Pa) for each level
PRESSURE_WEIGHTS_PATH = None  # '/mydata/deepcloud/yves/pressure_weights.npy'

# Output paths
OUTPUT_DIR = '/mydata/deepcloud/yves/results_final/metrics_tendency'
OUTPUT_FILENAME = '1D_2D_3D_GNN_models.md'

# Define target names and units
target_units = {
    "Total Temperature": "K s-1", 
    "Dynamical Temperature": "K s-1",
    "Zonal Wind ": "m s-2",
    "Meridional Wind": "m s-2",
    "Convective Humidity": "kg m-3 s-1",
    "Convective Cloud Water": "kg m-3 s-1",
    "Convective Cloud Ice": "kg m-3 s-1"
}

# Baseline model for skill score calculation
# Can be 'persistence', 'climatology', or the name of one of your models
BASELINE_FOR_SKILL = 'climatology'  # Use training mean as baseline
# Alternative: BASELINE_FOR_SKILL = 'GNN-1D-64-L2-100'  # Use 1D model as baseline

# ====================================================================
# DATA LOADING SECTION
# ====================================================================

def load_training_statistics(stats_path):
    """Load pre-computed training statistics for NRMSE normalization."""
    try:
        with open(stats_path, 'rb') as f:
            stats_dict = pickle.load(f)
        
        print(f"Loaded training statistics from: {stats_path}")
        print(f"Statistics keys: {stats_dict.keys()}")
        
        # Extract variance and compute std
        if 'train_target_variance' in stats_dict:
            train_variance = stats_dict['train_target_variance']
            train_std = np.sqrt(train_variance)
            print(f"Computed std from variance, shape: {train_std.shape}")
            
            # Also get mean if available (for climatology baseline)
            train_mean = None
            if 'train_target_mean' in stats_dict:
                train_mean = stats_dict['train_target_mean']
                print(f"Also loaded training mean, shape: {train_mean.shape}")
            
            # Print additional info if available
            if 'statistics_type' in stats_dict:
                print(f"Statistics type: {stats_dict['statistics_type']}")
            if 'n_files' in stats_dict:
                print(f"Computed from {stats_dict['n_files']} files")
            if 'subsample_rate' in stats_dict:
                print(f"Subsample rate: {stats_dict['subsample_rate']}")
            
            # Verify the shape matches expected number of targets
            if train_std.shape[-1] != 7:
                print(f"WARNING: Expected 7 target variables, but got {train_std.shape[-1]}")
                print("Check that the statistics were computed with the same target selection as training!")
                
            return train_std, train_mean
        else:
            print("Error: 'train_target_variance' not found in statistics file")
            return None, None
            
    except FileNotFoundError:
        print(f"Warning: Training statistics file not found at {stats_path}")
        print("NRMSE will be computed using test set std (not recommended)")
        return None, None

def load_model_predictions(model_path):
    """Load y_true and y_pred for a single model."""
    print(f'Loading test files from: {model_path}')
    
    # Load the data
    with open(join(model_path, 'y_true.pickle'), 'rb') as handle:
        y_true = pickle.load(handle)
    
    with open(join(model_path, 'y_pred.pickle'), 'rb') as handle:
        y_pred = pickle.load(handle)
    
    # Reshape if needed (assuming your data might be flattened)
    if isinstance(y_true, np.ndarray) and y_true.shape[-1] == 490:
        y_true = y_true.reshape(-1, 70, 7)
        y_pred = y_pred.reshape(-1, 70, 7)
    
    print(f'Data shapes - y_true: {y_true.shape}, y_pred: {y_pred.shape}')
    return y_true, y_pred

# ====================================================================
# METRICS CALCULATION SECTION
# ====================================================================

def calculate_metrics(y_true, y_pred, train_std=None, train_mean=None):
    """Calculate performance metrics between true and predicted values.
    
    Args:
        y_true: True values from test set
        y_pred: Predicted values from model
        train_std: Standard deviation from training set (for NRMSE). If None, uses test set std.
        train_mean: Mean from training set (for climatology baseline). Optional.
    
    Returns:
        Dictionary with metrics:
        - MAE: Mean Absolute Error
        - RMSE: Root Mean Squared Error
        - NRMSE: Normalized RMSE (using training std if provided)
        - nMAE: Normalized MAE (using training std if provided)
        - R2: Coefficient of determination (using test set mean)
        - Pearson_r: Pearson correlation coefficient (pattern agreement only)
    """
    # Convert to torch tensors if needed
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
    
    # Get original shapes
    original_shape = y_true.shape
    print(f"Original data shape: {original_shape}")
    
    # Reshape to 2D for easier computation - combine samples and heights
    if len(original_shape) > 2:
        # Shape is [batch, height, features] -> reshape to [batch*height, features]
        y_true_2d = y_true.reshape(-1, original_shape[-1])
        y_pred_2d = y_pred.reshape(-1, original_shape[-1])
    else:
        y_true_2d = y_true
        y_pred_2d = y_pred
    
    print(f"Reshaped for computation: {y_true_2d.shape}")
    
    # Calculate metrics per target variable
    metrics = {}
    
    # Mean Absolute Error
    mae = torch.mean(torch.abs(y_true_2d - y_pred_2d), dim=0)
    metrics['MAE'] = mae
    
    # Root Mean Squared Error
    mse = torch.mean((y_true_2d - y_pred_2d) ** 2, dim=0)
    rmse = torch.sqrt(mse)
    metrics['RMSE'] = rmse
    
    # Get normalization factor (training std)
    if train_std is not None:
        # Use training set standard deviation (recommended)
        if isinstance(train_std, np.ndarray):
            train_std = torch.tensor(train_std, dtype=torch.float32)
        
        # If train_std is per-level, we need to handle the reshaping
        if len(train_std.shape) > 1:
            # Reshape train_std to match the flattened structure
            train_std_2d = train_std.reshape(-1, train_std.shape[-1])
            # Take mean across levels to get per-variable std
            sigma = torch.mean(train_std_2d, dim=0)
        else:
            sigma = train_std
        
        print("Using training set std for normalization")
    else:
        # Fallback: use test set standard deviation (not recommended)
        sigma = torch.std(y_true_2d, dim=0)
        print("Warning: Using test set std for normalization (not recommended)")
    
    # Normalized RMSE (using training std)
    # For atmospheric variables, std should never be zero
    nrmse = rmse / sigma
    metrics['NRMSE'] = nrmse
    
    # Normalized MAE (using training std)
    nmae = mae / sigma
    metrics['nMAE'] = nmae
    
    # Also calculate test-normalized metrics
    # This uses the variance of the actual test subset being evaluated
    test_std = torch.std(y_true_2d, dim=0)
    nrmse_test = rmse / test_std
    nmae_test = mae / test_std
    metrics['NRMSE_test'] = nrmse_test
    metrics['nMAE_test'] = nmae_test
    
    # Log if test std differs significantly from training std
    std_ratio = test_std / sigma
    print(f"Test/Train std ratio: min={std_ratio.min():.2f}, max={std_ratio.max():.2f}")
    
    # R-squared - coefficient of determination (uses test set mean as required)
    y_true_mean = torch.mean(y_true_2d, dim=0)
    ss_tot = torch.sum((y_true_2d - y_true_mean.unsqueeze(0)) ** 2, dim=0)
    ss_res = torch.sum((y_true_2d - y_pred_2d) ** 2, dim=0)
    # For real test data, ss_tot should never be zero (would mean no variance)
    r2 = 1 - (ss_res / ss_tot)
    metrics['R2'] = r2
    
    # Pearson correlation coefficient (pattern agreement only, no bias)
    # This measures how well the patterns match, independent of systematic shifts
    pearson_r = torch.zeros(original_shape[-1])
    for i in range(original_shape[-1]):
        # Center the data (remove mean)
        y_true_centered = y_true_2d[:, i] - torch.mean(y_true_2d[:, i])
        y_pred_centered = y_pred_2d[:, i] - torch.mean(y_pred_2d[:, i])
        
        # Calculate correlation
        numerator = torch.sum(y_true_centered * y_pred_centered)
        denominator = torch.sqrt(torch.sum(y_true_centered ** 2) * torch.sum(y_pred_centered ** 2))
        # For real atmospheric data with variance, denominator should not be zero
        pearson_r[i] = numerator / denominator
    
    metrics['Pearson_r'] = pearson_r
    
    # Store MSE for skill score calculation later
    metrics['MSE'] = mse
    
    return metrics

def calculate_skill_scores(model_results, baseline_name, y_true=None, train_mean=None):
    """Calculate skill scores relative to a baseline model.
    
    Skill Score = 1 - (MSE_model / MSE_baseline)
    
    Positive values indicate improvement over baseline.
    SS = 0.3 means 30% reduction in MSE compared to baseline.
    
    Args:
        model_results: Dictionary of model results
        baseline_name: Either 'climatology' or name of a model
        y_true: True values (needed if baseline is climatology)
        train_mean: Training mean (needed if baseline is climatology)
    """
    skill_scores = {}
    
    if baseline_name == 'climatology':
        # Calculate MSE of climatology baseline (predicting training mean)
        if y_true is None or train_mean is None:
            print("Error: Need y_true and train_mean for climatology baseline")
            return skill_scores
            
        # Convert to tensors if needed
        if isinstance(y_true, np.ndarray):
            y_true = torch.tensor(y_true, dtype=torch.float32)
        if isinstance(train_mean, np.ndarray):
            train_mean = torch.tensor(train_mean, dtype=torch.float32)
            
        # Reshape y_true to 2D
        original_shape = y_true.shape
        if len(original_shape) > 2:
            y_true_2d = y_true.reshape(-1, original_shape[-1])
        else:
            y_true_2d = y_true
            
        # If train_mean is per-level, need to handle reshaping
        if len(train_mean.shape) > 1:
            # Broadcast train_mean to match y_true shape
            train_mean_broadcast = train_mean.unsqueeze(0).expand(original_shape[0], -1, -1)
            train_mean_2d = train_mean_broadcast.reshape(-1, original_shape[-1])
        else:
            # Single value per variable
            train_mean_2d = train_mean.unsqueeze(0).expand_as(y_true_2d)
            
        # Calculate climatology MSE
        baseline_mse = torch.mean((y_true_2d - train_mean_2d) ** 2, dim=0)
        print("Using climatology (training mean) as baseline for skill scores")
    else:
        # Use another model as baseline
        if baseline_name not in model_results:
            print(f"Warning: Baseline model '{baseline_name}' not found in results")
            return skill_scores
        baseline_mse = model_results[baseline_name]['MSE']
        print(f"Using {baseline_name} as baseline for skill scores")
    
    # Calculate skill scores for each model
    for model_name, metrics in model_results.items():
        model_mse = metrics['MSE']
        # For real data, baseline MSE should not be zero
        ss = 1 - (model_mse / baseline_mse)
        skill_scores[model_name] = ss
    
    return skill_scores

# ====================================================================
# RESULTS FORMATTING SECTION
# ====================================================================

def format_df_values(df, metric_name):
    """Format DataFrame values properly."""
    formatted_df = pd.DataFrame(index=df.index, columns=df.columns)
    
    if metric_name in ['R2', 'Pearson_r', 'Skill_Score']:
        # For correlation/skill metrics, format with 4 decimal places
        for col in df.columns:
            formatted_df[col] = df[col].map(lambda x: f"{x:.4f}")
    else:
        # For error metrics, use scientific notation
        for col in df.columns:
            formatted_df[col] = df[col].map(lambda x: f"{x:.4e}")
    
    return formatted_df

def get_formatted_table(df, metric_name, highlight_best=True):
    """Create a formatted table with the best performer highlighted."""
    formatted_df = format_df_values(df, metric_name)
    
    # Create list to hold formatted rows
    rows = []
    
    # Add header row
    headers = ['Target Variable'] + list(formatted_df.columns)
    rows.append(headers)
    
    # Add separator row
    rows.append(['---'] * len(headers))
    
    # Add data rows with highlighting
    for target in formatted_df.index:
        row = [target]
        values = df.loc[target]
        
        # Find the best value
        if metric_name in ['R2', 'Pearson_r', 'Skill_Score']:
            # Higher is better
            best_value = values.max()
        else:
            # Lower is better
            best_value = values.min()
        
        for model_name in formatted_df.columns:
            value = df.loc[target, model_name]
            formatted_value = formatted_df.loc[target, model_name]
            
            # Add highlighting for best value
            if highlight_best:
                if metric_name in ['R2', 'Pearson_r', 'Skill_Score']:
                    if value == best_value:
                        row.append(f"**{formatted_value}**")
                    else:
                        row.append(formatted_value)
                else:
                    if value == best_value:
                        row.append(f"**{formatted_value}**")
                    else:
                        row.append(formatted_value)
            else:
                row.append(formatted_value)
        
        rows.append(row)
    
    # Convert to a string table using tabulate
    table = tabulate(rows, tablefmt="pipe", headers="firstrow")
    return table

def create_summary_text(dataframes, metric_names, baseline_name):
    """Create the markdown summary text."""
    
    output_text = f"""# Model Performance Comparison

This summary shows performance metrics for all models, calculated over all test samples and all height levels.

## Recommended Minimal Metrics Set

Following best practices for emulator evaluation, we report:

1. **nRMSE** - Error magnitude relative to natural variability
2. **Pearson's r** - Pattern agreement (correlation)
3. **Skill Score** - Performance relative to baseline ({baseline_name})

## All Metrics Explanation

### Error Magnitude Metrics
- **MAE (Mean Absolute Error)**: Average absolute difference between predictions and true values (lower is better)
  - Units are the same as the target variables
  
- **RMSE (Root Mean Squared Error)**: Square root of average squared differences (lower is better)
  - Penalizes larger errors more than MAE
  - Units are the same as the target variables

- **nMAE (Normalized MAE)**: MAE divided by the standard deviation of training data (lower is better)
  - Dimensionless metric, allows comparison across different variables
  - Values < 1.0 indicate useful predictive skill

- **NRMSE (Normalized RMSE)**: RMSE divided by the standard deviation of training data (lower is better)
  - Measures error relative to the natural variation in the training data
  - Values < 1.0 indicate the model performs better than simply predicting the training mean
  - Dimensionless metric, allows comparison across different variables

- **nMAE_test / NRMSE_test**: Same as above but normalized by test subset standard deviation
  - Shows error relative to variability in the actual test samples being evaluated
  - Useful for understanding performance on this specific test subset
  - May differ from training-normalized values if test subset has different variability

### Pattern Agreement Metrics
- **R² (Coefficient of Determination)**: Proportion of test set variance explained by the model (higher is better)
  - R² = 1: Perfect prediction
  - R² = 0: Model predicts only the test set mean
  - R² < 0: Model performs worse than predicting the test set mean
  - Includes effects of both correlation and bias

- **Pearson r (Correlation Coefficient)**: Measures pattern agreement only (higher is better)
  - r = 1: Perfect positive correlation
  - r = 0: No linear correlation
  - r = -1: Perfect negative correlation
  - Independent of systematic bias - focuses only on pattern matching

### Skill Metrics
- **Skill Score**: Relative improvement over baseline model (higher is better)
  - SS = 1 - (MSE_model / MSE_baseline)
  - SS = 0.3 means 30% reduction in MSE compared to baseline
  - SS < 0 means worse than baseline
  - Baseline: {baseline_name}

*Note: Best values in each row are highlighted in bold.*

"""

    # Add each metric's table
    for metric_name in metric_names:
        output_text += f"\n## {metric_name}\n\n"
        
        # Add specific guidance for each metric
        if metric_name in ['R2', 'Pearson_r', 'Skill_Score']:
            output_text += "Higher values are better.\n\n"
        else:
            output_text += "Lower values are better.\n\n"
        
        output_text += get_formatted_table(dataframes[metric_name], metric_name) + "\n"

    # Add interpretation section
    output_text += """
## Key Insights

### Recommended Metrics Summary

The three recommended metrics provide complementary information:
- **nRMSE**: Overall error magnitude relative to natural variability
- **Pearson's r**: How well patterns are captured (independent of any systematic shifts)
- **Skill Score**: Clear improvement statement over baseline

### Interpreting Metric Combinations

1. **High R² but High NRMSE**: Model captures patterns well but has wrong magnitude (scaling issue)
2. **High Pearson r but Low R²**: Model gets patterns right but may have systematic offset
3. **Low NRMSE but Low R²**: Small errors but poor pattern capture (might be predicting mean)
4. **Good Skill Score**: Clear evidence of improvement over simpler approach

Note: For atmospheric tendency variables, absolute values are often more meaningful than relative percentages since tendency means are typically very close to zero.

"""

    return output_text

# ====================================================================
# MAIN EXECUTION
# ====================================================================

def main():
    print("Starting model performance evaluation with recommended metrics...")
    
    # Load training statistics for NRMSE normalization
    train_std, train_mean = load_training_statistics(TRAINING_STATS_PATH)
    
    # Store results for each model
    model_results = {}
    
    # Process each model
    print(f"\nCalculating metrics for {len(models)} models...")
    for model in models:
        print(f"\n--- Processing {model['name']} ---")
        
        # Load predictions
        y_true, y_pred = load_model_predictions(model['path'])
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred, train_std, train_mean)
        model_results[model['name']] = metrics
        
        print(f"Metrics calculated for {model['name']}")
    
    # Calculate skill scores
    print(f"\nCalculating skill scores relative to baseline: {BASELINE_FOR_SKILL}")
    
    # Need to keep y_true for climatology baseline
    # Use the last loaded y_true (they should all be the same)
    skill_scores = calculate_skill_scores(
        model_results, 
        BASELINE_FOR_SKILL,
        y_true=y_true if BASELINE_FOR_SKILL == 'climatology' else None,
        train_mean=train_mean if BASELINE_FOR_SKILL == 'climatology' else None
    )
    
    # Add skill scores to results
    for model_name in model_results:
        if model_name in skill_scores:
            model_results[model_name]['Skill_Score'] = skill_scores[model_name]
    
    # Define metrics to include in output (removed bias metrics)
    metric_names = [
        'nMAE', 'MAE', 'NRMSE', 'RMSE', 
        'nMAE_test', 'NRMSE_test',  # Add test-normalized metrics
        'Pearson_r', 'R2',
        'Skill_Score'
    ]
    
    target_names = list(target_units.keys())
    model_names = [model['name'] for model in models]
    
    print(f"\nCreating summary tables...")
    
    # Create DataFrames for each metric
    dataframes = {}
    for metric_name in metric_names:
        if metric_name in ['Skill_Score'] and metric_name not in model_results[model_names[0]]:
            print(f"Skipping {metric_name} - not calculated")
            continue
            
        data = {}
        for model_name, metrics in model_results.items():
            if metric_name in metrics:
                data[model_name] = metrics[metric_name].numpy()
        
        if data:  # Only create DataFrame if we have data
            df = pd.DataFrame(data, index=target_names)
            dataframes[metric_name] = df
    
    # Create summary text
    output_text = create_summary_text(dataframes, list(dataframes.keys()), BASELINE_FOR_SKILL)
    
    # Save results
    output_path = join(OUTPUT_DIR, OUTPUT_FILENAME)
    with open(output_path, 'w') as f:
        f.write(output_text)
    
    print(f"\nAnalysis complete!")
    print(f"Results saved to: {output_path}")
    
    # Print quick summary of recommended metrics
    print("\n=== Quick Summary of Recommended Metrics ===")
    print(f"Baseline for skill score: {BASELINE_FOR_SKILL}")
    
    for model_name in model_names:
        print(f"\n{model_name}:")
        if 'NRMSE' in model_results[model_name]:
            nrmse_mean = model_results[model_name]['NRMSE'].mean().item()
            print(f"  Average nRMSE: {nrmse_mean:.3f}")
        if 'NRMSE_test' in model_results[model_name]:
            nrmse_test_mean = model_results[model_name]['NRMSE_test'].mean().item()
            print(f"  Average nRMSE_test: {nrmse_test_mean:.3f}")
        if 'Pearson_r' in model_results[model_name]:
            pearson_mean = model_results[model_name]['Pearson_r'].mean().item()
            print(f"  Average Pearson r: {pearson_mean:.3f}")
        if 'Skill_Score' in model_results[model_name]:
            skill_mean = model_results[model_name]['Skill_Score'].mean().item()
            print(f"  Average Skill Score: {skill_mean:.3f}")

if __name__ == "__main__":
    main() 