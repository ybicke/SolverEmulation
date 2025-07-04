import torch
import pickle
import numpy as np
import pandas as pd
from os.path import join
from tabulate import tabulate
import os

# ====================================================================
# CONFIGURATION SECTION - Modify paths and settings here
# ====================================================================

# Flux models to compare
models = [
    # {'name': 'GNN-64-l3', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results/gnn_64_l3/test'},
    #{'name': 'GNN-128-l4', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results/gnn_128_l4/test'},
    #{'name': 'GNN-512-l3', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results/gnn_medium/test'},
    
    
    #{'name': 'ViT-128-l4', 'path': '/mydata/deepcloud/yves/results_git/vit_column_128_l4_h6_concat/test'},
    #{'name': 'AFNO-128-l4', 'path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_clean/test'},
    #{'name': 'BiLSTM-medium', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results/rnn_medium/test'},
    
    
    {'name': 'GNN-128-l4', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results/gnn_128_l4/test'},
    {'name': 'ViT-128-l4', 'path': '/mydata/deepcloud/yves/results_git/vit_column_128_l4_h6_concat/test'},
    {'name': 'AFNO-128-l4', 'path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_clean/test'},
    {'name': 'BiLSTM', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results/rnn_medium_second/test'},

    # {'name': 'BiLSTM-medium', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results/rnn_medium/test'},
    
    
    
    # Fluxes 1D models hrlu
    #{'name': 'AFNO-128-l4', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results/afno_1d_128_hrlu/test'},
    #{'name': 'ViT-128-l4', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results/vit_128_hrlu_0005_new/test'},
    #{'name': 'GNN-128-l4', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results/gnn_1d_128_hrlu_0005/test'},
    #{'name': 'BiLSTM-medium', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results/rnn_medium_hrlu/test'},
    #{'name': 'BiLSTM', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results/rnn_medium_hrlu_second/test'},

    
    ]

# Path to training statistics (for NRMSE normalization)
# NOTE: Currently commented out as training statistics not available for flux data
# TRAINING_STATS_PATH = '/mydata/deepcloud/yves/flux_training_statistics.pickle'
TRAINING_STATS_PATH = None

# Output paths
OUTPUT_DIR = '/mydata/deepcloud/yves/results_final/metrics_flux'
OUTPUT_FILENAME = 'Flux_1D_second.md'

# Make sure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define flux target names and units
flux_target_units = {
    "LW Upward Flux": "W m⁻²",
    "LW Downward Flux": "W m⁻²", 
    "SW Upward Flux": "W m⁻²",
    "SW Downward Flux": "W m⁻²"
}

# Define heating rate target names and units
heating_target_units = {
    "LW Heating Rate": "K day⁻¹",
    "SW Heating Rate": "K day⁻¹"
}

# Baseline model for skill score calculation
# Options: 'climatology' (requires training mean), 'persistence', or model name
# BASELINE_FOR_SKILL = 'climatology'  # Commented out - no training mean available
BASELINE_FOR_SKILL = 'BiLSTM-medium'  # Use simplest model as baseline

# ====================================================================
# DATA LOADING SECTION
# ====================================================================

def load_training_statistics(stats_path):
    """Load pre-computed training statistics for NRMSE normalization."""
    if stats_path is None:
        print("Training statistics path not provided - will use test set std for normalization")
        return None, None
        
    try:
        with open(stats_path, 'rb') as f:
            stats_dict = pickle.load(f)
        
        print(f"Loaded training statistics from: {stats_path}")
        print(f"Statistics keys: {stats_dict.keys()}")
        
        # Extract variance and compute std for both flux and heating data
        flux_std, heating_std = None, None
        flux_mean, heating_mean = None, None
        
        if 'train_flux_variance' in stats_dict:
            flux_variance = stats_dict['train_flux_variance']
            flux_std = np.sqrt(flux_variance)
            print(f"Computed flux std from variance, shape: {flux_std.shape}")
            
        if 'train_heating_variance' in stats_dict:
            heating_variance = stats_dict['train_heating_variance']
            heating_std = np.sqrt(heating_variance)
            print(f"Computed heating std from variance, shape: {heating_std.shape}")
            
        if 'train_flux_mean' in stats_dict:
            flux_mean = stats_dict['train_flux_mean']
            print(f"Loaded flux training mean, shape: {flux_mean.shape}")
            
        if 'train_heating_mean' in stats_dict:
            heating_mean = stats_dict['train_heating_mean']
            print(f"Loaded heating training mean, shape: {heating_mean.shape}")
            
        return (flux_std, heating_std), (flux_mean, heating_mean)
            
    except FileNotFoundError:
        print(f"Warning: Training statistics file not found at {stats_path}")
        print("Will use test set std for normalization (not recommended)")
        return (None, None), (None, None)

def load_flux_model_predictions(model_path):
    """Load flux and heating rate predictions for a single model."""
    print(f'Loading test files from: {model_path}')
    
    # Load flux data (y)
    with open(join(model_path, 'y_true.pickle'), 'rb') as handle:
        y_true = pickle.load(handle)
    
    with open(join(model_path, 'y_pred.pickle'), 'rb') as handle:
        y_pred = pickle.load(handle)
    
    # Load heating rate data (h)
    with open(join(model_path, 'h_true.pickle'), 'rb') as handle:
        h_true = pickle.load(handle)
        
    with open(join(model_path, 'h_pred.pickle'), 'rb') as handle:
        h_pred = pickle.load(handle)
    
    # Convert to tensors if needed
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
    if isinstance(h_true, np.ndarray):
        h_true = torch.tensor(h_true, dtype=torch.float32)
    if isinstance(h_pred, np.ndarray):
        h_pred = torch.tensor(h_pred, dtype=torch.float32)
    
    print(f'Data shapes - y_true: {y_true.shape}, y_pred: {y_pred.shape}')
    print(f'Data shapes - h_true: {h_true.shape}, h_pred: {h_pred.shape}')
    
    return (y_true, y_pred), (h_true, h_pred)

# ====================================================================
# METRICS CALCULATION SECTION
# ====================================================================

def calculate_flux_metrics(y_true, y_pred, train_std=None, train_mean=None):
    """Calculate performance metrics for flux or heating rate predictions.
    
    Args:
        y_true: True values from test set
        y_pred: Predicted values from model
        train_std: Standard deviation from training set (for NRMSE). If None, uses test set std.
        train_mean: Mean from training set (for climatology baseline). Optional.
    
    Returns:
        Dictionary with metrics for each variable
    """
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
    nrmse = rmse / sigma
    metrics['NRMSE'] = nrmse
    
    # Normalized MAE (using training std)
    nmae = mae / sigma
    metrics['nMAE'] = nmae
    
    # Also calculate test-normalized metrics
    test_std = torch.std(y_true_2d, dim=0)
    nrmse_test = rmse / test_std
    nmae_test = mae / test_std
    metrics['NRMSE_test'] = nrmse_test
    metrics['nMAE_test'] = nmae_test
    
    # Log if test std differs significantly from training std
    if train_std is not None:
        std_ratio = test_std / sigma
        print(f"Test/Train std ratio: min={std_ratio.min():.2f}, max={std_ratio.max():.2f}")
        
    # R-squared - coefficient of determination (uses test set mean as required)
    y_true_mean = torch.mean(y_true_2d, dim=0)
    ss_tot = torch.sum((y_true_2d - y_true_mean.unsqueeze(0)) ** 2, dim=0)
    ss_res = torch.sum((y_true_2d - y_pred_2d) ** 2, dim=0)
    r2 = 1 - (ss_res / ss_tot)
    metrics['R2'] = r2
    
    # Pearson correlation coefficient
    pearson_r = torch.zeros(original_shape[-1])
    for i in range(original_shape[-1]):
        # Center the data
        y_true_centered = y_true_2d[:, i] - torch.mean(y_true_2d[:, i])
        y_pred_centered = y_pred_2d[:, i] - torch.mean(y_pred_2d[:, i])
        
        # Calculate correlation
        numerator = torch.sum(y_true_centered * y_pred_centered)
        denominator = torch.sqrt(torch.sum(y_true_centered ** 2) * torch.sum(y_pred_centered ** 2))
        pearson_r[i] = numerator / denominator
    
    metrics['Pearson_r'] = pearson_r
    
    # Store MSE for skill score calculation later
    metrics['MSE'] = mse
    
    return metrics

def calculate_skill_scores(model_results, baseline_name, y_true=None, train_mean=None):
    """Calculate skill scores relative to a baseline model."""
    skill_scores = {}
    
    if baseline_name == 'climatology':
        if y_true is None or train_mean is None:
            print("Error: Need y_true and train_mean for climatology baseline")
            print("Skipping skill score calculation")
            return skill_scores
            
        # Calculate MSE of climatology baseline
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
            
        # Handle train_mean broadcasting
        if len(train_mean.shape) > 1:
            train_mean_broadcast = train_mean.unsqueeze(0).expand(original_shape[0], -1, -1)
            train_mean_2d = train_mean_broadcast.reshape(-1, original_shape[-1])
        else:
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
            if highlight_best and value == best_value:
                        row.append(f"**{formatted_value}**")
            else:
                row.append(formatted_value)
        
        rows.append(row)
    
    # Convert to a string table using tabulate
    table = tabulate(rows, tablefmt="pipe", headers="firstrow")
    return table

def create_flux_summary_text(combined_dataframes, _, metric_names, baseline_name):
    """Create the markdown summary text for flux models."""
    
    output_text = f"""# Flux Model Performance Comparison

This summary shows performance metrics for radiative flux and heating rate models, calculated over all test samples and all height levels.

## Data Overview

The evaluation includes two types of radiative transfer predictions:

### Radiative Fluxes (4 variables, 71 height levels):
- **LW Upward Flux**: Longwave radiation moving upward [W m⁻²]
- **LW Downward Flux**: Longwave radiation moving downward [W m⁻²]  
- **SW Upward Flux**: Shortwave radiation moving upward [W m⁻²]
- **SW Downward Flux**: Shortwave radiation moving downward [W m⁻²]

### Heating Rates (2 variables, 70 height levels):
- **LW Heating Rate**: Longwave radiative heating rate [K day⁻¹]
- **SW Heating Rate**: Shortwave radiative heating rate [K day⁻¹]

## Recommended Minimal Metrics Set

Following best practices for emulator evaluation, we report:

1. **nRMSE** - Error magnitude relative to natural variability
2. **Pearson's r** - Pattern agreement (correlation)
3. **Skill Score** - Performance relative to baseline ({baseline_name})

## All Metrics Explanation

### Error Magnitude Metrics
- **MAE (Mean Absolute Error)**: Average absolute difference between predictions and true values (lower is better)
- **RMSE (Root Mean Squared Error)**: Square root of average squared differences (lower is better)
- **nMAE (Normalized MAE)**: MAE divided by standard deviation (lower is better)
- **NRMSE (Normalized RMSE)**: RMSE divided by standard deviation (lower is better)
- **nMAE_test / NRMSE_test**: Same as above but normalized by test subset standard deviation

### Pattern Agreement Metrics
- **R² (Coefficient of Determination)**: Proportion of variance explained by the model (higher is better)
- **Pearson r (Correlation Coefficient)**: Measures pattern agreement only (higher is better)

### Skill Metrics
- **Skill Score**: Relative improvement over baseline model (higher is better)
  - SS = 1 - (MSE_model / MSE_baseline)
  - Baseline: {baseline_name}

*Note: Best values in each row are highlighted in bold.*

"""

    # Add combined results (flux and heating rates in same tables)
    output_text += "\n# COMBINED FLUX AND HEATING RATE RESULTS\n\n"
    for metric_name in metric_names:
        if metric_name in combined_dataframes:
            output_text += f"\n## {metric_name}\n\n"
            if metric_name in ['R2', 'Pearson_r', 'Skill_Score']:
                output_text += "Higher values are better.\n\n"
            else:
                output_text += "Lower values are better.\n\n"
            output_text += get_formatted_table(combined_dataframes[metric_name], metric_name) + "\n"

    # Add interpretation section
    output_text += f"""
## Key Insights

### Radiative Transfer Context

Radiative fluxes and heating rates are fundamental to atmospheric physics:
- **Fluxes**: Energy transport through radiation (W m⁻²)
- **Heating rates**: Local energy deposition causing temperature changes (K day⁻¹)

### Model Performance Interpretation

For radiative transfer emulation:
1. **Flux accuracy** affects energy balance calculations
2. **Heating rate accuracy** affects temperature tendency predictions
3. **Vertical profile accuracy** is crucial for atmospheric stability

### Baseline Comparison

Skill scores are calculated relative to **{baseline_name}** model.
- Positive skill scores indicate improvement over the baseline
- Values > 0.2 typically indicate meaningful improvement
- Values < 0 indicate worse performance than baseline

### Physical Relevance

- **SW fluxes/heating**: Strongly dependent on solar zenith angle, clouds, aerosols
- **LW fluxes/heating**: Primarily controlled by temperature and water vapor profiles
- **Vertical structure**: Critical for representing atmospheric radiative cooling/heating

"""

    return output_text

# ====================================================================
# MAIN EXECUTION
# ====================================================================

def main():
    print("Starting flux model performance evaluation...")
    
    # Load training statistics for NRMSE normalization
    if TRAINING_STATS_PATH:
        (flux_train_std, heating_train_std), (flux_train_mean, heating_train_mean) = load_training_statistics(TRAINING_STATS_PATH)
    else:
        print("No training statistics path provided - using test set normalization")
        flux_train_std, heating_train_std = None, None
        flux_train_mean, heating_train_mean = None, None
    
    # Store results for each model
    flux_model_results = {}
    heating_model_results = {}
    
    # Process each model
    print(f"\nCalculating metrics for {len(models)} models...")
    for model in models:
        print(f"\n--- Processing {model['name']} ---")
        
        # Load predictions
        (y_true, y_pred), (h_true, h_pred) = load_flux_model_predictions(model['path'])
        
        # Calculate flux metrics
        print("Calculating flux metrics...")
        flux_metrics = calculate_flux_metrics(y_true, y_pred, flux_train_std, flux_train_mean)
        flux_model_results[model['name']] = flux_metrics
        
        # Calculate heating rate metrics
        print("Calculating heating rate metrics...")
        heating_metrics = calculate_flux_metrics(h_true, h_pred, heating_train_std, heating_train_mean)
        heating_model_results[model['name']] = heating_metrics
        
        print(f"Metrics calculated for {model['name']}")
    
    # Calculate skill scores if baseline is available
    print(f"\nCalculating skill scores relative to baseline: {BASELINE_FOR_SKILL}")
    
    if BASELINE_FOR_SKILL == 'climatology':
        if flux_train_mean is None or heating_train_mean is None:
            print("Warning: Cannot calculate climatology skill scores - no training means available")
            print("Skipping skill score calculation")
            flux_skill_scores = {}
            heating_skill_scores = {}
        else:
            flux_skill_scores = calculate_skill_scores(
                flux_model_results, BASELINE_FOR_SKILL, y_true, flux_train_mean
            )
            heating_skill_scores = calculate_skill_scores(
                heating_model_results, BASELINE_FOR_SKILL, h_true, heating_train_mean
            )
    else:
        # Use another model as baseline
        flux_skill_scores = calculate_skill_scores(flux_model_results, BASELINE_FOR_SKILL)
        heating_skill_scores = calculate_skill_scores(heating_model_results, BASELINE_FOR_SKILL)
    
    # Add skill scores to results
    for model_name in flux_model_results:
        if model_name in flux_skill_scores:
            flux_model_results[model_name]['Skill_Score'] = flux_skill_scores[model_name]
        if model_name in heating_skill_scores:
            heating_model_results[model_name]['Skill_Score'] = heating_skill_scores[model_name]
    
    # Define metrics to include in output
    metric_names = [
        'nMAE', 'MAE', 'NRMSE', 'RMSE', 
        'nMAE_test', 'NRMSE_test',
        'Pearson_r', 'R2',
        'Skill_Score'
    ]
    
    flux_target_names = list(flux_target_units.keys())
    heating_target_names = list(heating_target_units.keys())
    model_names = [model['name'] for model in models]
    
    print(f"\nCreating summary tables...")
    
    # Create combined DataFrames for both flux and heating rate metrics
    combined_dataframes = {}
    for metric_name in metric_names:
        if metric_name == 'Skill_Score' and not flux_skill_scores:
            continue
            
        # Combine flux and heating rate data
        data = {}
        for model_name, metrics in flux_model_results.items():
            if metric_name in metrics:
                data[model_name] = metrics[metric_name].numpy()
        
        # Add heating rate data to the same table
        heating_data = {}
        for model_name, metrics in heating_model_results.items():
            if metric_name in metrics:
                heating_data[model_name] = metrics[metric_name].numpy()
        
        if data and heating_data:
            # Create combined DataFrame with both flux and heating targets
            all_target_names = flux_target_names + heating_target_names
            
            # Stack flux and heating data
            combined_data = {}
            for model_name in data.keys():
                if model_name in heating_data:
                    combined_values = np.concatenate([data[model_name], heating_data[model_name]])
                    combined_data[model_name] = combined_values
            
            if combined_data:
                df = pd.DataFrame(combined_data, index=all_target_names)
                combined_dataframes[metric_name] = df
    
    # Create summary text
    output_text = create_flux_summary_text(
        combined_dataframes, combined_dataframes, 
        list(combined_dataframes.keys()), BASELINE_FOR_SKILL
    )
    
    # Save results
    output_path = join(OUTPUT_DIR, OUTPUT_FILENAME)
    with open(output_path, 'w') as f:
        f.write(output_text)
    
    print(f"\nAnalysis complete!")
    print(f"Results saved to: {output_path}")
    
    # Print quick summary
    print("\n=== Quick Summary of Recommended Metrics ===")
    print(f"Baseline for skill score: {BASELINE_FOR_SKILL}")
    
    print("\nFLUX MODELS:")
    for model_name in model_names:
        print(f"\n{model_name}:")
        if 'NRMSE' in flux_model_results[model_name]:
            nrmse_mean = flux_model_results[model_name]['NRMSE'].mean().item()
            print(f"  Flux Average nRMSE: {nrmse_mean:.3f}")
        if 'NRMSE_test' in flux_model_results[model_name]:
            nrmse_test_mean = flux_model_results[model_name]['NRMSE_test'].mean().item()
            print(f"  Flux Average nRMSE_test: {nrmse_test_mean:.3f}")
        if 'Pearson_r' in flux_model_results[model_name]:
            pearson_mean = flux_model_results[model_name]['Pearson_r'].mean().item()
            print(f"  Flux Average Pearson r: {pearson_mean:.3f}")
        if 'Skill_Score' in flux_model_results[model_name]:
            skill_mean = flux_model_results[model_name]['Skill_Score'].mean().item()
            print(f"  Flux Average Skill Score: {skill_mean:.3f}")
    
    print("\nHEATING RATE MODELS:")
    for model_name in model_names:
        print(f"\n{model_name}:")
        if 'NRMSE' in heating_model_results[model_name]:
            nrmse_mean = heating_model_results[model_name]['NRMSE'].mean().item()
            print(f"  Heating Average nRMSE: {nrmse_mean:.3f}")
        if 'NRMSE_test' in heating_model_results[model_name]:
            nrmse_test_mean = heating_model_results[model_name]['NRMSE_test'].mean().item()
            print(f"  Heating Average nRMSE_test: {nrmse_test_mean:.3f}")
        if 'Pearson_r' in heating_model_results[model_name]:
            pearson_mean = heating_model_results[model_name]['Pearson_r'].mean().item()
            print(f"  Heating Average Pearson r: {pearson_mean:.3f}")
        if 'Skill_Score' in heating_model_results[model_name]:
            skill_mean = heating_model_results[model_name]['Skill_Score'].mean().item()
            print(f"  Heating Average Skill Score: {skill_mean:.3f}")

if __name__ == "__main__":
    main() 