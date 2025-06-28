import torch
import pickle
import numpy as np
import pandas as pd
from os.path import join
import os

# ====================================================================
# CONFIGURATION SECTION
# ====================================================================

# Models to evaluate
models = [
    {'name': 'GNN-128-l4', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results/gnn_128_l4/test'},
    #{'name': 'ViT-128-l4', 'path': '/mydata/deepcloud/yves/results_git/vit_column_128_l4_h6_concat/test'},
    #{'name': 'AFNO-128-l4', 'path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_clean/test'},
    {'name': 'BiLSTM-medium', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results/rnn_medium/test'},
]

# Output settings
OUTPUT_DIR = '/mydata/deepcloud/yves/results_final/Flux_Metrics'
OUTPUT_FILENAME = 'simple_mae_summary.md'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Variable names (order matches data indices)
flux_names = [
    "LW Upward Flux",      # y[:, 0]
    "LW Downward Flux",    # y[:, 1]
    "SW Upward Flux",      # y[:, 2]
    "SW Downward Flux"     # y[:, 3]
]

heating_names = [
    "LW Heating Rate",     # h[:, 0]
    "SW Heating Rate"      # h[:, 1]
]

all_variable_names = flux_names + heating_names

# ====================================================================
# MAIN CALCULATION
# ====================================================================

def load_and_calculate_mae(model_path):
    """Load model predictions and calculate MAE for all variables."""
    print(f'Loading data from: {model_path}')
    
    # Load flux data
    with open(join(model_path, 'y_true.pickle'), 'rb') as f:
        y_true = pickle.load(f)
    with open(join(model_path, 'y_pred.pickle'), 'rb') as f:
        y_pred = pickle.load(f)
    
    # Load heating rate data
    with open(join(model_path, 'h_true.pickle'), 'rb') as f:
        h_true = pickle.load(f)
    with open(join(model_path, 'h_pred.pickle'), 'rb') as f:
        h_pred = pickle.load(f)
    
    # Convert to tensors
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true, dtype=torch.float32)
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
    if isinstance(h_true, np.ndarray):
        h_true = torch.tensor(h_true, dtype=torch.float32)
        h_pred = torch.tensor(h_pred, dtype=torch.float32)
    
    print(f'  Flux shapes: y_true={y_true.shape}, y_pred={y_pred.shape}')
    print(f'  Heating shapes: h_true={h_true.shape}, h_pred={h_pred.shape}')
    
    # Calculate MAE across samples and height levels (average over all spatial points)
    # For flux data: calculate MAE for each of the 4 variables
    flux_mae = torch.mean(torch.abs(y_true - y_pred), dim=(0, 1))  # Average over samples and height levels
    
    # For heating rate data: calculate MAE for each of the 2 variables  
    heating_mae = torch.mean(torch.abs(h_true - h_pred), dim=(0, 1))  # Average over samples and height levels
    
    # Combine all MAE values
    all_mae = torch.cat([flux_mae, heating_mae])
    
    print(f'  Calculated MAE for {len(all_mae)} variables')
    return all_mae.numpy()

def main():
    print("=" * 60)
    print("SIMPLE MAE CALCULATION FOR FLUX MODELS")
    print("=" * 60)
    
    # Store results
    results = {}
    
    # Calculate MAE for each model
    for model in models:
        print(f"\nProcessing {model['name']}...")
        mae_values = load_and_calculate_mae(model['path'])
        results[model['name']] = mae_values
    
    # Create DataFrame
    df = pd.DataFrame(results, index=all_variable_names)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    # Print results to console
    print("\nMAE Results (averaged across all samples and height levels):")
    print("-" * 60)
    for var_name in all_variable_names:
        print(f"\n{var_name}:")
        for model_name in df.columns:
            mae_val = df.loc[var_name, model_name]
            if var_name in flux_names:
                print(f"  {model_name:15}: {mae_val:.3f} W m⁻²")
            else:
                print(f"  {model_name:15}: {mae_val:.3f} K day⁻¹")
    
    # Calculate and print averages
    print(f"\n{'OVERALL AVERAGES'}")
    print("-" * 60)
    
    # Flux average
    flux_avg = df.loc[flux_names].mean()
    print(f"\nAverage Flux MAE (4 variables):")
    for model_name in df.columns:
        print(f"  {model_name:15}: {flux_avg[model_name]:.3f} W m⁻²")
    
    # Heating average  
    heating_avg = df.loc[heating_names].mean()
    print(f"\nAverage Heating Rate MAE (2 variables):")
    for model_name in df.columns:
        print(f"  {model_name:15}: {heating_avg[model_name]:.3f} K day⁻¹")
    
    # Overall average
    overall_avg = df.mean()
    print(f"\nOverall Average MAE (all 6 variables):")
    for model_name in df.columns:
        print(f"  {model_name:15}: {overall_avg[model_name]:.3f}")
    
    # Model ranking
    print(f"\nModel Ranking (by overall average MAE, lower is better):")
    ranked = overall_avg.sort_values()
    for i, (model_name, mae_val) in enumerate(ranked.items(), 1):
        print(f"  {i}. {model_name:15}: {mae_val:.3f}")
    
    # Create markdown output
    markdown_content = f"""# Simple MAE Summary for Flux Models

## Overview
Mean Absolute Error (MAE) calculated across all test samples and all height levels.

## Results by Variable

| Variable | Unit | {' | '.join(df.columns)} |
|:---------|:-----|{':---:|' * len(df.columns)}
"""
    
    for var_name in all_variable_names:
        unit = "W m⁻²" if var_name in flux_names else "K day⁻¹"
        row_values = [f"{df.loc[var_name, col]:.3f}" for col in df.columns]
        
        # Highlight best performance
        min_val = min([df.loc[var_name, col] for col in df.columns])
        highlighted_values = []
        for col in df.columns:
            val = df.loc[var_name, col]
            if val == min_val:
                highlighted_values.append(f"**{val:.3f}**")
            else:
                highlighted_values.append(f"{val:.3f}")
        
        markdown_content += f"| {var_name} | {unit} | {' | '.join(highlighted_values)} |\n"
    
    # Add summary section
    markdown_content += f"""
## Summary Statistics

### Average by Category

| Category | {' | '.join(df.columns)} |
|:---------|{':---:|' * len(df.columns)}
"""
    
    # Flux averages
    flux_avg_values = [f"{flux_avg[col]:.3f}" for col in df.columns]
    min_flux = min([flux_avg[col] for col in df.columns])
    highlighted_flux = [f"**{flux_avg[col]:.3f}**" if flux_avg[col] == min_flux else f"{flux_avg[col]:.3f}" for col in df.columns]
    markdown_content += f"| Flux Average (W m⁻²) | {' | '.join(highlighted_flux)} |\n"
    
    # Heating averages
    heating_avg_values = [f"{heating_avg[col]:.3f}" for col in df.columns]
    min_heating = min([heating_avg[col] for col in df.columns])
    highlighted_heating = [f"**{heating_avg[col]:.3f}**" if heating_avg[col] == min_heating else f"{heating_avg[col]:.3f}" for col in df.columns]
    markdown_content += f"| Heating Average (K day⁻¹) | {' | '.join(highlighted_heating)} |\n"
    
    # Overall averages
    overall_avg_values = [f"{overall_avg[col]:.3f}" for col in df.columns]
    min_overall = min([overall_avg[col] for col in df.columns])
    highlighted_overall = [f"**{overall_avg[col]:.3f}**" if overall_avg[col] == min_overall else f"{overall_avg[col]:.3f}" for col in df.columns]
    markdown_content += f"| Overall Average | {' | '.join(highlighted_overall)} |\n"
    
    # Add ranking
    markdown_content += f"""
### Model Ranking (Overall Average MAE)

"""
    for i, (model_name, mae_val) in enumerate(ranked.items(), 1):
        markdown_content += f"{i}. **{model_name}**: {mae_val:.3f}\n"
    
    markdown_content += """
*Note: Lower MAE values indicate better performance. Best values in each row are highlighted in bold.*
"""
    
    # Save results
    output_path = join(OUTPUT_DIR, OUTPUT_FILENAME)
    with open(output_path, 'w') as f:
        f.write(markdown_content)
    
    print(f"\n" + "=" * 60)
    print(f"Results saved to: {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    main() 