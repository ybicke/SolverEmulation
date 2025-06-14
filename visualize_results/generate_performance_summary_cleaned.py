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
    {'name': 'GNN-3D-64-L2-100', 'path': '/mydata/deepcloud/yves/results-temp/gnn3d_id39_tendency_64_l2_100/test'},
    {'name': 'GNN-1D-64-L2-100', 'path': '/mydata/deepcloud/yves/results-temp/gnn_64_l2_tendency_1d_triangle_fully_connected/test'},
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
OUTPUT_DIR = '/mydata/deepcloud/yves/final_results'
OUTPUT_FILENAME = '3D_vs_1D_tendency_eval_cleaned.md'

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
                
            return train_std
        else:
            print("Error: 'train_target_variance' not found in statistics file")
            return None
            
    except FileNotFoundError:
        print(f"Warning: Training statistics file not found at {stats_path}")
        print("NRMSE will be computed using test set std (not recommended)")
        return None

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

def calculate_metrics(y_true, y_pred, train_std=None):
    """Calculate performance metrics between true and predicted values.
    
    Args:
        y_true: True values from test set
        y_pred: Predicted values from model
        train_std: Standard deviation from training set (for NRMSE). If None, uses test set std.
    
    Returns:
        Dictionary with metrics:
        - MAE: Mean Absolute Error
        - RMSE: Root Mean Squared Error
        - NRMSE: Normalized RMSE (using training std if provided)
        - R2: Coefficient of determination (using test set mean)
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
    
    # Normalized RMSE
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
        
        print("Using training set std for NRMSE normalization")
    else:
        # Fallback: use test set standard deviation (not recommended)
        sigma = torch.std(y_true_2d, dim=0)
        print("Warning: Using test set std for NRMSE normalization (not recommended)")
    
    # Avoid division by zero
    epsilon = 1e-8
    nrmse = rmse / (sigma + epsilon)
    metrics['NRMSE'] = nrmse
    
    # R-squared - coefficient of determination (uses test set mean as required)
    y_true_mean = torch.mean(y_true_2d, dim=0)  # Mean of the test set being evaluated
    ss_tot = torch.sum((y_true_2d - y_true_mean.unsqueeze(0)) ** 2, dim=0)
    ss_res = torch.sum((y_true_2d - y_pred_2d) ** 2, dim=0)
    r2 = 1 - (ss_res / (ss_tot + epsilon))  # Add epsilon to avoid division by zero
    metrics['R2'] = r2
    
    return metrics

# ====================================================================
# RESULTS FORMATTING SECTION
# ====================================================================

def format_df_values(df, metric_name):
    """Format DataFrame values properly."""
    formatted_df = pd.DataFrame(index=df.index, columns=df.columns)
    
    if metric_name == 'R2':
        # For R2, format with 4 decimal places
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
        
        # Find the best value (higher for R2, lower for error metrics)
        if metric_name == 'R2':
            best_value = values.max()
        else:
            best_value = values.min()
        
        for model_name in formatted_df.columns:
            value = df.loc[target, model_name]
            formatted_value = formatted_df.loc[target, model_name]
            
            # Add highlighting for best value
            if highlight_best and ((metric_name == 'R2' and value == best_value) or 
                                 (metric_name != 'R2' and value == best_value)):
                row.append(f"**{formatted_value}**")
            else:
                row.append(formatted_value)
        
        rows.append(row)
    
    # Convert to a string table using tabulate
    table = tabulate(rows, tablefmt="pipe", headers="firstrow")
    return table

def create_summary_text(dataframes, metric_names):
    """Create the markdown summary text."""
    
    output_text = """# Model Performance Comparison

This summary shows performance metrics for all models, calculated over all test samples and all height levels.

## Metrics Explanation

- **MAE (Mean Absolute Error)**: Average absolute difference between predictions and true values (lower is better)
  - Units are the same as the target variables
  
- **RMSE (Root Mean Squared Error)**: Square root of average squared differences (lower is better)
  - Penalizes larger errors more than MAE
  - Units are the same as the target variables

- **NRMSE (Normalized RMSE)**: RMSE divided by the standard deviation of training data (lower is better)
  - Measures error relative to the natural variation in the training data
  - Values < 1.0 indicate the model performs better than simply predicting the training mean
  - Dimensionless metric, allows comparison across different variables
  
- **R² (Coefficient of Determination)**: Proportion of test set variance explained by the model (higher is better)
  - R² = 1: Perfect prediction
  - R² = 0: Model predicts only the test set mean
  - R² < 0: Model performs worse than predicting the test set mean

*Note: Best values in each row are highlighted in bold.*

"""

    # Add each metric's table
    for metric_name in metric_names:
        output_text += f"\n## {metric_name}\n\n"
        if metric_name == 'R2':
            output_text += "Higher values are better.\n\n"
        else:
            output_text += "Lower values are better.\n\n"
        
        output_text += get_formatted_table(dataframes[metric_name], metric_name) + "\n"

    return output_text

# ====================================================================
# MAIN EXECUTION
# ====================================================================

def main():
    print("Starting model performance evaluation...")
    
    # Load training statistics for NRMSE normalization
    train_std = load_training_statistics(TRAINING_STATS_PATH)
    
    # Store results for each model
    model_results = {}
    
    # Process each model
    print(f"\nCalculating metrics for {len(models)} models...")
    for model in models:
        print(f"\n--- Processing {model['name']} ---")
        
        # Load predictions
        y_true, y_pred = load_model_predictions(model['path'])
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred, train_std)
        model_results[model['name']] = metrics
        
        print(f"Metrics calculated for {model['name']}")
    
    # Convert results to pandas DataFrames
    metric_names = ['MAE', 'RMSE', 'NRMSE', 'R2']
    target_names = list(target_units.keys())
    model_names = [model['name'] for model in models]
    
    print(f"\nCreating summary tables...")
    
    # Create DataFrames for each metric
    dataframes = {}
    for metric_name in metric_names:
        data = {}
        for model_name, metrics in model_results.items():
            data[model_name] = metrics[metric_name].numpy()
        
        df = pd.DataFrame(data, index=target_names)
        dataframes[metric_name] = df
    
    # Create summary text
    output_text = create_summary_text(dataframes, metric_names)
    
    # Save results
    output_path = join(OUTPUT_DIR, OUTPUT_FILENAME)
    with open(output_path, 'w') as f:
        f.write(output_text)
    
    print(f"\nAnalysis complete!")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main() 