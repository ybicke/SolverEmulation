import torch
import pickle
import numpy as np
import pandas as pd
from os.path import join
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

# Models to compare
models = [
    #{'name': 'GNN-3D-64-L2-100','path': '/mydata/deepcloud/yves/results-temp/gnn3d_id39_tendency_64_l2_100/test'},
    #{'name': 'GNN-1D-64-L2-100','path': '/mydata/deepcloud/yves/results-temp/gnn_64_l2_tendency_1d_triangle_fully_connected/test'},
    
    {'name': 'GNN-1D-64-L2-100', 'path': '/mydata/deepcloud/yves/results-new/gnn_1d_tendency_64_l2_100_k4_correct_variance/test'},
    {'name': 'GNN-3D-64-L2-100', 'path': '/mydata/deepcloud/yves/results-new/gnn_3d_tendency_64_l2_100_k4_correct_variance/test'},
]

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

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    """Calculate performance metrics between true and predicted values.
    
    Metrics are calculated over ALL samples and ALL height levels,
    giving a single metric value for each target variable (7 in total).
    
    Returns:
        Dictionary with metrics:
        - NRMSE: Normalized RMSE (RMSE / (max - min)) - error as fraction of data range
        - RelRMSE: Relative RMSE (RMSE / mean of true values) - how large errors are relative to true values
        - MAPE: Mean Absolute Percentage Error - average % error (where safe to calculate)
        - R2: Coefficient of determination (how well the model explains variance)
    """
    # Convert to torch tensors if needed
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred)
    
    # Get original shapes
    original_shape = y_true.shape
    
    # Reshape to 2D for easier computation - combine samples and heights
    if len(original_shape) > 2:
        # Shape is [batch, height, features] -> reshape to [batch*height, features]
        y_true_2d = y_true.reshape(-1, original_shape[-1])
        y_pred_2d = y_pred.reshape(-1, original_shape[-1])
    else:
        y_true_2d = y_true
        y_pred_2d = y_pred
    
    # Calculate global metrics per target variable
    metrics = {}
    
    # Mean Absolute Error - for reference, but won't be primarily used
    mae = torch.mean(torch.abs(y_true_2d - y_pred_2d), dim=0)
    
    # Root Mean Squared Error
    mse = torch.mean((y_true_2d - y_pred_2d) ** 2, dim=0)
    rmse = torch.sqrt(mse)
    
    # Use standard deviation instead of range for normalization (more robust to outliers)
    # This measures error relative to the natural variation in the data
    y_std = torch.std(y_true_2d, dim=0)
    # Avoid division by zero by adding a small epsilon where std is close to zero
    epsilon = 1e-8
    nrmse = rmse / (y_std + epsilon)
    metrics['NRMSE'] = nrmse
    
    # Relative RMSE - RMSE divided by the mean absolute value of y_true
    # Expresses error relative to the magnitude of the values being predicted
    y_true_abs_mean = torch.mean(torch.abs(y_true_2d), dim=0)
    rel_rmse = rmse / (y_true_abs_mean + epsilon)
    metrics['RelRMSE'] = rel_rmse
    
    # Mean Absolute Percentage Error (MAPE)
    # Calculate only for non-zero true values to avoid division by zero
    # This expresses error as a percentage of the true value
    abs_diff = torch.abs(y_true_2d - y_pred_2d)
    abs_true = torch.abs(y_true_2d)
    
    # Create a mask for non-zero true values (using a slightly higher threshold)
    nonzero_mask = abs_true > 1e-5  # Filter out very small values that lead to excessive MAPE
    
    # Initialize MAPE as tensor of zeros
    mape = torch.zeros(y_true_2d.shape[1], device=y_true_2d.device)
    
    # Calculate MAPE for each feature
    for i in range(y_true_2d.shape[1]):
        feature_mask = nonzero_mask[:, i]
        if torch.any(feature_mask):
            # Calculate MAPE only for non-zero values
            feature_mape = torch.mean(abs_diff[feature_mask, i] / abs_true[feature_mask, i])
            mape[i] = feature_mape * 100  # Convert to percentage
    
    metrics['MAPE'] = mape
    
    # R-squared - coefficient of determination
    y_true_mean = torch.mean(y_true_2d, dim=0)
    ss_tot = torch.sum((y_true_2d - y_true_mean.unsqueeze(0)) ** 2, dim=0)
    ss_res = torch.sum((y_true_2d - y_pred_2d) ** 2, dim=0)
    r2 = 1 - (ss_res / ss_tot)
    metrics['R2'] = r2
    
    return metrics

# Store results for each model
model_results = {}

# Process each model
print("Calculating metrics for all models...")
for model in models:
    test_path = model['path']
    print(f'Loading test files... ({test_path})')
    
    # Load the data
    with open(join(test_path, 'y_true.pickle'), 'rb') as handle:
        y_true = pickle.load(handle)
    
    with open(join(test_path, 'y_pred.pickle'), 'rb') as handle:
        y_pred = pickle.load(handle)
    
    # Reshape if needed
    if isinstance(y_true, np.ndarray) and y_true.shape[-1] == 490:
        y_true = y_true.reshape(-1, 70, 7)
        y_pred = y_pred.reshape(-1, 70, 7)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    model_results[model['name']] = metrics

# Convert results to pandas DataFrames for easier visualization
metric_names = ['NRMSE', 'RelRMSE', 'MAPE', 'R2']  # Focusing on relative metrics
target_names = list(target_units.keys())
model_names = [model['name'] for model in models]

# Create DataFrames for each metric
dataframes = {}
for metric_name in metric_names:
    # Create DataFrame
    data = {}
    for model_name, metrics in model_results.items():
        data[model_name] = metrics[metric_name].numpy()
    
    df = pd.DataFrame(data, index=target_names)
    dataframes[metric_name] = df

# Fix DataFrame.applymap deprecation warnings
def format_df_values(df, metric_name):
    """Format DataFrame values properly, replacing the deprecated applymap."""
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
    """
    Create a formatted table with the best performer highlighted.
    For R2, higher is better. For error metrics, lower is better.
    """
    # Make a copy for formatting
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
        
        # Find the best value (min or max depending on metric)
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

# Create a summary text file with the performance tables
output_text = """# Model Performance Comparison

This summary shows performance metrics for all models, calculated over all samples and all height levels.

## Metrics Explanation

- **NRMSE (Normalized RMSE)**: RMSE divided by the standard deviation of true values (lower is better)
  - Measures error relative to the natural variation in the data
  - Values of 1.0 mean prediction errors are about the same size as natural variations
  - Values < 1.0 indicate the model performs better than simply predicting the mean
  
- **RelRMSE (Relative RMSE)**: RMSE divided by the mean absolute value of true data (lower is better)
  - Expresses error relative to the magnitude of the values being predicted
  - Values of 1.0 mean errors are roughly the same size as the values themselves
  - Values > 1.0 mean errors are larger than the true values on average

- **MAPE (Mean Absolute Percentage Error)**: Average percentage error (lower is better)
  - Values represent the average percentage difference between predictions and true values
  - Very small true values are excluded to avoid excessive percentages

- **R² (Coefficient of Determination)**: Proportion of variance explained by the model (higher is better)
  - R² = 1: Perfect prediction
  - R² = 0: Model predicts only the mean value
  - R² < 0: Model performs worse than predicting the mean

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

# Add a ranking summary
output_text += "\n## Model Ranking (Average across NRMSE and RelRMSE)\n\n"
output_text += "Lower rank is better (1 = best).\n\n"

# Create a ranking DataFrame (1 = best, 4 = worst)
ranking = pd.DataFrame(index=target_names, columns=model_names)

for target in target_names:
    for metric in ['NRMSE', 'RelRMSE']:  # Use relative metrics for ranking
        # Get values for this target and metric
        values = dataframes[metric].loc[target]
        
        # Convert to ranks (1 = best/lowest error)
        sorted_idx = np.argsort(values)
        ranks = np.empty_like(sorted_idx)
        ranks[sorted_idx] = np.arange(1, len(values) + 1)
        
        # Add these ranks to the existing values (or initialize if first metric)
        if metric == 'NRMSE':
            ranking.loc[target] = ranks
        else:
            ranking.loc[target] += ranks

# Lower total rank is better (ranked better across metrics)
# Divide by 2 to get average rank across the 2 metrics
ranking = ranking / 2

# Format the ranking table
# Use DataFrame.map instead of applymap (which is deprecated)
ranking_formatted = pd.DataFrame(index=ranking.index, columns=ranking.columns)
for col in ranking.columns:
    ranking_formatted[col] = ranking[col].map(lambda x: f"{x:.1f}")

rows = []
headers = ['Target Variable'] + list(ranking.columns)
rows.append(headers)
rows.append(['---'] * len(headers))

for target in ranking.index:
    row = [target]
    values = ranking.loc[target]
    best_rank = values.min()
    
    for model_name in ranking.columns:
        rank = ranking.loc[target, model_name]
        formatted_rank = ranking_formatted.loc[target, model_name]
        
        # Add highlighting for best rank
        if rank == best_rank:
            row.append(f"**{formatted_rank}**")
        else:
            row.append(formatted_rank)
    
    rows.append(row)

# Add ranking table to output text
output_text += tabulate(rows, tablefmt="pipe", headers="firstrow") + "\n"

# Add an overall winner summary
output_text += "\n## Overall Best Model Per Target Variable\n\n"

best_models = []
for target in target_names:
    avg_rank = ranking.loc[target]
    best_model = avg_rank.idxmin()  # Get model with lowest average rank
    best_rank = avg_rank.min()
    
    best_models.append([target, best_model, f"{best_rank:.1f}"])

# Create the best models table
best_models_headers = ["Target Variable", "Best Model", "Average Rank"]
best_models_table = tabulate([best_models_headers, ['---']*3] + best_models, tablefmt="pipe", headers="firstrow")

output_text += best_models_table

# Save the text file
with open('/mydata/deepcloud/yves/final_results/3D_vs_1D_tendency_eval_old.md', 'w') as f:
    f.write(output_text)

# Create a heatmap of the ranking (for visual comparison) - fixed for proper numeric data
try:
    # Convert ranking to float type to ensure it works with heatmap
    ranking_numeric = ranking.astype(float)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(ranking_numeric, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                linewidths=.5, cbar_kws={'label': 'Average Rank (lower is better)'})
    plt.title('Model Ranking by Target Variable (lower is better)', fontsize=16)
    plt.tight_layout()
    plt.savefig('/mydata/deepcloud/yves/final_results/3D_vs_1D_tendency_eval.png', 
                bbox_inches='tight', dpi=300)
    print("Ranking heatmap saved successfully.")
except Exception as e:
    print(f"Error generating heatmap: {e}")

print("\nAnalysis complete! Results saved to:")
print("1. Markdown summary: /mydata/deepcloud/yves/results-temp/model_performance_new.md")
print("2. Ranking heatmap: /mydata/deepcloud/yves/results-temp/model_ranking.png")
print("\nYou can view the markdown file directly in VS Code or any markdown viewer.") 