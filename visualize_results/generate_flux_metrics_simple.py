import torch
import pickle
import numpy as np
import pandas as pd
from os.path import join
import os

# ====================================================================
# CONFIGURATION
# ====================================================================

models = [
    {'name': 'GNN-128-l4', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results/gnn_128_l4/test'},
    {'name': 'ViT-128-l4', 'path': '/mydata/deepcloud/yves/results_git/vit_column_128_l4_h6_concat/test'},
    {'name': 'AFNO-128-l4', 'path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_clean/test'},
    {'name': 'BiLSTM-medium', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results/rnn_medium/test'},
]

OUTPUT_DIR = '/mydata/deepcloud/yves/results_final/Flux_Metrics'
OUTPUT_FILENAME = 'flux_metrics_simple.md'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Variable names (must match data order)
flux_names = ["LW Upward Flux", "LW Downward Flux", "SW Upward Flux", "SW Downward Flux"]
heating_names = ["LW Heating Rate", "SW Heating Rate"]
all_names = flux_names + heating_names

# ====================================================================
# CORE FUNCTIONS
# ====================================================================

def load_model_data(model_path):
    """Load flux and heating data for one model."""
    print(f'Loading: {model_path}')
    
    # Load flux data
    with open(join(model_path, 'y_true.pickle'), 'rb') as f:
        y_true = torch.tensor(pickle.load(f), dtype=torch.float32)
    with open(join(model_path, 'y_pred.pickle'), 'rb') as f:
        y_pred = torch.tensor(pickle.load(f), dtype=torch.float32)
    
    # Load heating data
    with open(join(model_path, 'h_true.pickle'), 'rb') as f:
        h_true = torch.tensor(pickle.load(f), dtype=torch.float32)
    with open(join(model_path, 'h_pred.pickle'), 'rb') as f:
        h_pred = torch.tensor(pickle.load(f), dtype=torch.float32)
    
    print(f'  Flux: {y_true.shape}, Heating: {h_true.shape}')
    return (y_true, y_pred), (h_true, h_pred)

def calculate_metrics(y_true, y_pred):
    """Calculate core metrics for predictions."""
    # Flatten to [samples*heights, variables] for computation
    y_true_flat = y_true.reshape(-1, y_true.shape[-1])
    y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
    
    # Calculate metrics per variable
    mae = torch.mean(torch.abs(y_true_flat - y_pred_flat), dim=0)
    mse = torch.mean((y_true_flat - y_pred_flat) ** 2, dim=0)
    rmse = torch.sqrt(mse)
    
    # R-squared
    y_mean = torch.mean(y_true_flat, dim=0)
    ss_tot = torch.sum((y_true_flat - y_mean) ** 2, dim=0)
    ss_res = torch.sum((y_true_flat - y_pred_flat) ** 2, dim=0)
    r2 = 1 - (ss_res / ss_tot)
    
    # Pearson correlation
    pearson_r = torch.zeros(y_true.shape[-1])
    for i in range(y_true.shape[-1]):
        y_true_center = y_true_flat[:, i] - torch.mean(y_true_flat[:, i])
        y_pred_center = y_pred_flat[:, i] - torch.mean(y_pred_flat[:, i])
        numerator = torch.sum(y_true_center * y_pred_center)
        denominator = torch.sqrt(torch.sum(y_true_center ** 2) * torch.sum(y_pred_center ** 2))
        pearson_r[i] = numerator / denominator
    
    return {
        'MAE': mae.numpy(),
        'RMSE': rmse.numpy(), 
        'R2': r2.numpy(),
        'Pearson_r': pearson_r.numpy()
    }

def format_table(df, metric_name):
    """Create a simple markdown table."""
    # Determine if higher or lower is better
    higher_better = metric_name in ['R2', 'Pearson_r']
    
    table = f"| Variable | {' | '.join(df.columns)} |\n"
    table += f"|:---------|{':---:|' * len(df.columns)}\n"
    
    for var_name in df.index:
        values = df.loc[var_name]
        best_val = values.max() if higher_better else values.min()
        
        row = f"| {var_name} |"
        for model in df.columns:
            val = df.loc[var_name, model]
            if metric_name in ['R2', 'Pearson_r']:
                formatted = f"{val:.4f}"
            else:
                formatted = f"{val:.3e}"
            
            # Bold if best
            if val == best_val:
                formatted = f"**{formatted}**"
            row += f" {formatted} |"
        table += row + "\n"
    
    return table

# ====================================================================
# MAIN EXECUTION
# ====================================================================

def main():
    print("=" * 50)
    print("SIMPLIFIED FLUX METRICS CALCULATION")
    print("=" * 50)
    
    # Store all results
    all_results = {}
    
    # Process each model
    for model in models:
        print(f"\nProcessing {model['name']}...")
        
        # Load data
        (y_true, y_pred), (h_true, h_pred) = load_model_data(model['path'])
        
        # Calculate metrics
        flux_metrics = calculate_metrics(y_true, y_pred)
        heating_metrics = calculate_metrics(h_true, h_pred)
        
        # Store combined results
        all_results[model['name']] = {
            'flux': flux_metrics,
            'heating': heating_metrics
        }
    
    # Create combined DataFrames
    print("\nCreating summary tables...")
    
    # Combine all metrics into single tables
    metric_dfs = {}
    
    for metric_name in ['MAE', 'RMSE', 'R2', 'Pearson_r']:
        data = {}
        for model_name, results in all_results.items():
            # Combine flux + heating values
            combined_values = np.concatenate([
                results['flux'][metric_name],
                results['heating'][metric_name]
            ])
            data[model_name] = combined_values
        
        df = pd.DataFrame(data, index=all_names)
        metric_dfs[metric_name] = df
    
    # Print console summary
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    
    for model_name in [m['name'] for m in models]:
        print(f"\n{model_name}:")
        overall_mae = metric_dfs['MAE'][model_name].mean()
        overall_r2 = metric_dfs['R2'][model_name].mean()
        overall_corr = metric_dfs['Pearson_r'][model_name].mean()
        print(f"  Average MAE: {overall_mae:.3e}")
        print(f"  Average R²: {overall_r2:.3f}")
        print(f"  Average Correlation: {overall_corr:.3f}")
    
    # Create markdown output
    content = """# Flux Model Performance Metrics

## Overview
Performance metrics for radiative flux and heating rate predictions across all test samples and height levels.

### Variables
- **Flux variables (4):** LW Upward/Downward, SW Upward/Downward [W m⁻²]
- **Heating variables (2):** LW/SW Heating Rates [K day⁻¹]

### Metrics
- **MAE:** Mean Absolute Error (lower is better)
- **RMSE:** Root Mean Squared Error (lower is better)  
- **R²:** Coefficient of determination (higher is better)
- **Pearson r:** Correlation coefficient (higher is better)

*Best values in each row are highlighted in bold.*

"""
    
    # Add metric tables
    for metric_name in ['MAE', 'RMSE', 'R2', 'Pearson_r']:
        content += f"\n## {metric_name}\n\n"
        if metric_name in ['R2', 'Pearson_r']:
            content += "Higher values are better.\n\n"
        else:
            content += "Lower values are better.\n\n"
        content += format_table(metric_dfs[metric_name], metric_name) + "\n"
    
    # Add summary table
    content += "\n## Overall Model Performance\n\n"
    content += "Average across all 6 variables.\n\n"
    
    summary_data = {}
    for model_name in [m['name'] for m in models]:
        summary_data[model_name] = [
            metric_dfs['MAE'][model_name].mean(),
            metric_dfs['RMSE'][model_name].mean(),
            metric_dfs['R2'][model_name].mean(),
            metric_dfs['Pearson_r'][model_name].mean()
        ]
    
    summary_df = pd.DataFrame(summary_data, index=['MAE', 'RMSE', 'R²', 'Pearson r'])
    
    content += "| Metric | " + " | ".join(summary_df.columns) + " |\n"
    content += "|:-------|" + ":---:|" * len(summary_df.columns) + "\n"
    
    for metric in summary_df.index:
        values = summary_df.loc[metric]
        best_val = values.max() if metric in ['R²', 'Pearson r'] else values.min()
        
        row = f"| {metric} |"
        for model in summary_df.columns:
            val = summary_df.loc[metric, model]
            if metric in ['R²', 'Pearson r']:
                formatted = f"{val:.4f}"
            else:
                formatted = f"{val:.3e}"
            
            if val == best_val:
                formatted = f"**{formatted}**"
            row += f" {formatted} |"
        content += row + "\n"
    
    # Save results
    output_path = join(OUTPUT_DIR, OUTPUT_FILENAME)
    with open(output_path, 'w') as f:
        f.write(content)
    
    print(f"\n{'='*50}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main() 