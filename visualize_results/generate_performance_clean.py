import torch
import pickle
import numpy as np
import pandas as pd
from os.path import join, exists
from tabulate import tabulate

# Models to compare
models = [
    #{'name': 'AFNO', 'path': '/mydata/deepcloud/yves/results-temp/afno_column_1percent_Emb128_clean_tendency_normTarg/test'},
    #{'name': 'GNN-32-L2','path': '/mydata/deepcloud/yves/results-temp/gnn_32_l2_tendency_normTarg/test'},
    #{'name': 'GNN-64-L2', 'path': '/mydata/deepcloud/yves/results-temp/gnn_64_l2_tendency_normTarg/test'},
    #{'name': 'GNN-128-L2', 'path': '/mydata/deepcloud/yves/results-temp/gnn_128_l2_tendency_normTarg/test'},
    
    
    {'name': 'GNN-3D-64-L2-100', 'path': '/mydata/deepcloud/yves/results-temp/gnn3d_id39_tendency_64_l2_100/test'},
    # {'name': 'GNN-3D-64-L2-100-Indep','path': '/mydata/deepcloud/yves/results-temp/gnn3d_id39_tendency_64_l2_indep_100/test'},
    {'name': 'GNN-1D-64-L2-100','path': '/mydata/deepcloud/yves/results-temp/gnn_64_l2_tendency_1d_triangle/test'},
]

# Define target names and units
target_units = {
    "Sum of Temperature Tendency": "K s⁻¹", 
    "Dynamical Temperature Tendency": "K s⁻¹",
    "Sum of Zonal Wind Tendency": "m s⁻²",
    "Sum of Meridional Wind Tendency": "m s⁻²",
    "Convective Tend. Absolute Humidity": "kg m⁻³ s⁻¹",
    "Convective Tend. Cloud Water Mass Density": "kg m⁻³ s⁻¹",
    "Convective Tend. Cloud Ice Mass Density": "kg m⁻³ s⁻¹"
}

def calculate_core_metrics(y_true, y_pred):
    """
    Calculate baseline-independent core metrics for physics emulation.
    These are the most important metrics and don't require external baselines.
    
    Args:
        y_true: True tendency values
        y_pred: Model predicted tendencies
    
    Returns:
        Dictionary with core physics metrics
    """
    # Convert to torch tensors if needed
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred)
    
    # Reshape to 2D for easier computation [samples, features]
    original_shape = y_true.shape
    if len(original_shape) > 2:
        y_true_2d = y_true.reshape(-1, original_shape[-1])
        y_pred_2d = y_pred.reshape(-1, original_shape[-1])
    else:
        y_true_2d = y_true
        y_pred_2d = y_pred
    
    metrics = {}
    
    # Calculate standard deviation first as we'll need it multiple times
    y_std = torch.std(y_true_2d, dim=0)
    
    # 1. R² - PRIMARY METRIC (uses mean of your test data)
    y_mean = torch.mean(y_true_2d, dim=0)
    ss_tot = torch.sum((y_true_2d - y_mean.unsqueeze(0)) ** 2, dim=0)
    ss_res = torch.sum((y_true_2d - y_pred_2d) ** 2, dim=0)
    r2 = 1 - (ss_res / ss_tot)
    metrics['R2'] = r2
    
    # 2. NRMSE - Scale-independent error (normalized by your test data std)
    mse_model = torch.mean((y_true_2d - y_pred_2d) ** 2, dim=0)
    rmse_model = torch.sqrt(mse_model)
    nrmse = rmse_model / y_std
    metrics['NRMSE'] = nrmse
    
    # 3. ACC - Pattern correlation (anomalies from your test data mean)
    y_true_anom = y_true_2d - torch.mean(y_true_2d, dim=0, keepdim=True)
    y_pred_anom = y_pred_2d - torch.mean(y_pred_2d, dim=0, keepdim=True)
    
    acc = torch.zeros(y_true_2d.shape[1])
    for i in range(y_true_2d.shape[1]):
        numerator = torch.sum(y_true_anom[:, i] * y_pred_anom[:, i])
        denominator = torch.sqrt(torch.sum(y_true_anom[:, i] ** 2) * torch.sum(y_pred_anom[:, i] ** 2))
        acc[i] = numerator / denominator
    metrics['ACC'] = acc
    
    # 4. MAE - Direct physical error in atmospheric units
    mae_model = torch.mean(torch.abs(y_true_2d - y_pred_2d), dim=0)
    metrics['MAE'] = mae_model
    
    # 4b. NMAE - Normalized Mean Absolute Error (scale-independent like NRMSE)
    nmae = mae_model / y_std  # Normalize by same std as NRMSE
    metrics['NMAE'] = nmae
    
    # 5. Additional useful baseline-independent metrics
    
    # Explained Variance (alternative to R²)
    explained_var = 1 - torch.var(y_true_2d - y_pred_2d, dim=0) / torch.var(y_true_2d, dim=0)
    metrics['Explained_Variance'] = explained_var
    
    # Pearson correlation coefficient
    corr_coeffs = torch.zeros(y_true_2d.shape[1])
    for i in range(y_true_2d.shape[1]):
        corr_matrix = torch.corrcoef(torch.stack([y_true_2d[:, i], y_pred_2d[:, i]]))
        corr_coeffs[i] = corr_matrix[0, 1]
    metrics['Pearson_Correlation'] = corr_coeffs
    
    return metrics

def calculate_baseline_metrics(y_true, y_pred, baseline_mean):
    """
    Calculate baseline-dependent metrics like Skill Score using train target mean baseline.
    
    Args:
        y_true: True tendency values
        y_pred: Model predicted tendencies
        baseline_mean: Train target mean baseline values
    
    Returns:
        Dictionary with baseline-dependent metrics
    """
    # Convert to torch tensors if needed
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred)
    if isinstance(baseline_mean, np.ndarray):
        baseline_mean = torch.tensor(baseline_mean)
    
    # Reshape to 2D
    original_shape = y_true.shape
    if len(original_shape) > 2:
        y_true_2d = y_true.reshape(-1, original_shape[-1])
        y_pred_2d = y_pred.reshape(-1, original_shape[-1])
    else:
        y_true_2d = y_true
        y_pred_2d = y_pred
    
    metrics = {}
    epsilon = 1e-10
    
    # Create baseline predictions using train target mean
    baseline_pred = baseline_mean.unsqueeze(0).expand_as(y_true_2d)
    
    # Calculate baseline errors
    mse_model = torch.mean((y_true_2d - y_pred_2d) ** 2, dim=0)
    rmse_model = torch.sqrt(mse_model)
    
    mse_baseline = torch.mean((y_true_2d - baseline_pred) ** 2, dim=0)
    rmse_baseline = torch.sqrt(mse_baseline)
    
    # Skill Score (Murphy, 1988)
    skill_score = 1 - (rmse_model / (rmse_baseline + epsilon))
    metrics['Skill_Score'] = skill_score
    
    # Relative improvement in MSE
    mse_improvement = (mse_baseline - mse_model) / (mse_baseline + epsilon)
    metrics['MSE_Improvement'] = mse_improvement    
    
    # Store baseline info
    metrics['Baseline_Type'] = "Train Target Mean"
    metrics['Baseline_RMSE'] = rmse_baseline
    metrics['Model_RMSE'] = rmse_model
    
    return metrics

def format_scientific_notation(value, precision=2):
    """Format small values in scientific notation for better readability."""
    if abs(value) < 1e-4:  # Use scientific notation for values smaller than 0.0001
        return f"{value:.{precision}e}"
    else:
        return f"{value:.{precision+2}f}"

# Configuration options
CALCULATE_BASELINE_METRICS = True  # Set to True when you have full training baseline
FULL_TRAINING_BASELINE_PATH = '/mydata/deepcloud/yves/h5_tendency_data_all/train_target_statistics.pickle'  # Path to full training baseline pickle file

# Store results for each model
model_results = {}

print("=== CORE PHYSICS METRICS EVALUATION ===")
print("Evaluating baseline-independent metrics for numerical solver emulation...")
print()

# Process each model
for model in models:
    test_path = model['path']
    print(f'Processing {model["name"]}... ({test_path})')
    
    # Load the data
    with open(join(test_path, 'y_true.pickle'), 'rb') as handle:
        y_true = pickle.load(handle)
    
    with open(join(test_path, 'y_pred.pickle'), 'rb') as handle:
        y_pred = pickle.load(handle)
    
    # Reshape if needed (handle both 3D and 4D formats)
    if isinstance(y_true, np.ndarray) and y_true.shape[-1] == 490:
        y_true = y_true.reshape(-1, 70, 7)
        y_pred = y_pred.reshape(-1, 70, 7)
    
    # Calculate core metrics (baseline-independent)
    core_metrics = calculate_core_metrics(y_true, y_pred)
    model_results[model['name']] = {'core': core_metrics}
    
    # Optionally calculate baseline-dependent metrics
    if CALCULATE_BASELINE_METRICS and FULL_TRAINING_BASELINE_PATH:
        if exists(FULL_TRAINING_BASELINE_PATH):
            with open(FULL_TRAINING_BASELINE_PATH, 'rb') as f:
                baseline_stats = pickle.load(f)
            print(f"  Using full training baseline from: {FULL_TRAINING_BASELINE_PATH}")
            
            # Handle different formats of baseline data
            if isinstance(baseline_stats, dict):
                # Extract mean from statistics dictionary
                if 'mean' in baseline_stats:
                    baseline_mean = baseline_stats['mean']
                elif 'target_mean' in baseline_stats:
                    baseline_mean = baseline_stats['target_mean']
                else:
                    # Print available keys to help debug
                    print(f"  Available keys in baseline file: {list(baseline_stats.keys())}")
                    # Try to find a key that might contain the mean
                    possible_mean_keys = [k for k in baseline_stats.keys() if 'mean' in k.lower()]
                    if possible_mean_keys:
                        baseline_mean = baseline_stats[possible_mean_keys[0]]
                        print(f"  Using key '{possible_mean_keys[0]}' as baseline mean")
                    else:
                        print(f"  ERROR: Could not find mean in baseline statistics")
                        continue
            else:
                # Assume it's directly the mean tensor/array
                baseline_mean = baseline_stats
            
            baseline_metrics = calculate_baseline_metrics(y_true, y_pred, baseline_mean)
            model_results[model['name']]['baseline'] = baseline_metrics
        else:
            print(f"  WARNING: Baseline file not found: {FULL_TRAINING_BASELINE_PATH}")

# Create comprehensive analysis
target_names = list(target_units.keys())
model_names = [model['name'] for model in models]

# Core metrics (most important, baseline-independent)
core_metric_names = ['R2', 'ACC', 'NRMSE', 'NMAE', 'MAE']

def create_clean_summary():
    output_text = """# Physics Emulation Model Evaluation

This analysis focuses on **baseline-independent core metrics** for numerical solver emulation.

## Evaluation Context

**Task**: Atmospheric numerical solver emulation (input physics → tendency output)  
**Data**: Aqua planet, 1 year, no seasonality  
**Training**: 10% subset sampling  
**Focus**: Instantaneous input-output physics relationships

## Core Metrics (Baseline-Independent)

These metrics use only your own data statistics and are the most important for physics emulation:

### R² (Coefficient of Determination) - PRIMARY METRIC
- **Physical meaning**: Fraction of atmospheric tendency variance explained by your model
- **Formula**: R² = 1 - SS_residual/SS_total (uses test data mean)
- **Range**: -∞ to 1.0 (1.0 = perfect physics, 0.0 = predicts mean, <0 = worse than mean)
- **Interpretation**: R² = 0.7 means you capture 70% of atmospheric physics variability

### ACC (Anomaly Correlation Coefficient) - PATTERN SKILL
- **Physical meaning**: How well you capture atmospheric patterns  
- **Formula**: Correlation of anomalies from test data mean
- **Range**: -1 to 1 (1 = perfect patterns, 0 = no pattern skill)
- **Interpretation**: Critical for tiny tendency values where patterns matter more than magnitude

### NRMSE (Normalized Root Mean Square Error) - SCALE-INDEPENDENT ERROR
- **Physical meaning**: Prediction error relative to natural atmospheric variability
- **Formula**: NRMSE = RMSE / σ_test_data  
- **Range**: 0 to ∞ (0 = perfect, <1.0 = better than natural variability)
- **Interpretation**: NRMSE = 0.3 means errors are 30% of natural variability in your test data

### NMAE (Normalized Mean Absolute Error) - SCALE-INDEPENDENT ERROR
- **Physical meaning**: Absolute error relative to natural atmospheric variability
- **Formula**: NMAE = MAE / σ_test_data  
- **Range**: 0 to ∞ (0 = perfect, <1.0 = better than natural variability)
- **Interpretation**: NMAE = 0.3 means absolute errors are 30% of natural variability

### MAE (Mean Absolute Error) - PHYSICAL UNITS
- **Physical meaning**: Average error in actual atmospheric units (K/s, m/s², etc.)
- **Range**: 0 to ∞ (0 = perfect)
- **Interpretation**: Directly interpretable in atmospheric physics terms
- **Note**: Values displayed in scientific notation for very small errors

"""

    # Create DataFrames for core metrics
    dataframes = {}
    for metric_name in core_metric_names:
        data = {}
        for model_name in model_names:
            data[model_name] = model_results[model_name]['core'][metric_name].numpy()
        
        df = pd.DataFrame(data, index=target_names)
        dataframes[metric_name] = df
    
    # Add overall averages
    for metric_name in core_metric_names:
        df = dataframes[metric_name]
        overall_mean = df.mean(axis=0)
        df.loc['OVERALL AVERAGE'] = overall_mean
    
    # Add individual metric tables
    for metric_name in core_metric_names:
        if metric_name == 'R2':
            output_text += f"\n## {metric_name} - Variance Explained (Higher = Better)\n\n"
        elif metric_name == 'ACC':
            output_text += f"\n## {metric_name} - Pattern Recognition (Higher = Better)\n\n"
        elif metric_name == 'NRMSE':
            output_text += f"\n## {metric_name} - Normalized Error (Lower = Better)\n\n"
        elif metric_name == 'NMAE':
            output_text += f"\n## {metric_name} - Normalized Error (Lower = Better)\n\n"
        elif metric_name == 'MAE':
            output_text += f"\n## {metric_name} - Physical Error (Lower = Better)\n\n"
        
        # Format table
        df = dataframes[metric_name]
        
        # Add highlighting for best values
        rows = []
        headers = ['Target Variable'] + list(df.columns)
        rows.append(headers)
        rows.append(['---'] * len(headers))
        
        for target in df.index:
            row = [target]
            values = df.loc[target]
            
            # Find best value
            if metric_name in ['R2', 'ACC']:
                best_value = values.max()
            else:
                best_value = values.min()
            
            for model_name in df.columns:
                value = df.loc[target, model_name]
                
                # Format values based on metric type
                if metric_name == 'MAE':
                    formatted_value = format_scientific_notation(value)
                else:
                    formatted_value = f"{value:.4f}"
                
                # Highlight best (except overall average)
                if (target != 'OVERALL AVERAGE' and 
                    ((metric_name in ['R2', 'ACC'] and value == best_value) or 
                     (metric_name in ['NRMSE', 'NMAE', 'MAE'] and value == best_value))):
                    row.append(f"**{formatted_value}**")
                else:
                    row.append(formatted_value)
            
            rows.append(row)
        
        table = tabulate(rows, tablefmt="pipe", headers="firstrow")
        output_text += table + "\n\n"
    
    # Add composite ranking
    output_text += """
## Composite Physics Score

Ranking based on three most important baseline-independent metrics:
- **R²** (40% weight) - Primary physics capture
- **ACC** (35% weight) - Pattern recognition  
- **NRMSE** (25% weight) - Scale-independent error

"""
    
    # Calculate composite scores
    composite_scores = pd.DataFrame(index=target_names, columns=model_names, dtype=float)
    
    for target in target_names:
        # Normalize metrics to 0-1 scale
        r2_vals = dataframes['R2'].loc[target]
        acc_vals = dataframes['ACC'].loc[target]
        nrmse_vals = dataframes['NRMSE'].loc[target]
        
        # Normalize (handle edge cases)
        r2_range = r2_vals.max() - r2_vals.min()
        acc_range = acc_vals.max() - acc_vals.min()
        nrmse_range = nrmse_vals.max() - nrmse_vals.min()
        
        if r2_range > 1e-10:
            r2_norm = (r2_vals - r2_vals.min()) / r2_range
        else:
            r2_norm = pd.Series([0.5] * len(r2_vals), index=r2_vals.index)
            
        if acc_range > 1e-10:
            acc_norm = (acc_vals - acc_vals.min()) / acc_range
        else:
            acc_norm = pd.Series([0.5] * len(acc_vals), index=acc_vals.index)
            
        if nrmse_range > 1e-10:
            nrmse_norm = (nrmse_vals.max() - nrmse_vals) / nrmse_range
        else:
            nrmse_norm = pd.Series([0.5] * len(nrmse_vals), index=nrmse_vals.index)
        
        # Weighted composite
        composite_scores.loc[target] = 0.40 * r2_norm + 0.35 * acc_norm + 0.25 * nrmse_norm
    
    # Add overall average
    overall_composite = composite_scores.mean(axis=0)
    composite_scores.loc['OVERALL AVERAGE'] = overall_composite
    
    # Format composite table
    rows = []
    headers = ['Target Variable'] + list(composite_scores.columns)
    rows.append(headers)
    rows.append(['---'] * len(headers))
    
    for target in composite_scores.index:
        row = [target]
        values = composite_scores.loc[target]
        best_score = values.max()
        
        for model_name in composite_scores.columns:
            score = composite_scores.loc[target, model_name]
            formatted_score = f"{score:.4f}"
            
            # Highlight best score
            if target != 'OVERALL AVERAGE' and abs(score - best_score) < 1e-6:
                row.append(f"**{formatted_score}**")
            else:
                row.append(formatted_score)
        
        rows.append(row)
    
    output_text += tabulate(rows, tablefmt="pipe", headers="firstrow") + "\n"
    
    # Add baseline metrics if calculated
    if CALCULATE_BASELINE_METRICS and 'baseline' in model_results[model_names[0]]:
        output_text += f"""

## Baseline-Dependent Metrics

*These metrics compare your model to train target mean baseline*

### Skill Score (Higher = Better)

Measures improvement over baseline: SS = 1 - RMSE_model/RMSE_baseline
- **SS > 0**: Model performs better than baseline
- **SS = 0**: Model performs same as baseline  
- **SS < 0**: Model performs worse than baseline

"""
        
        # Create skill score table
        skill_data = {}
        for model_name in model_names:
            skill_data[model_name] = model_results[model_name]['baseline']['Skill_Score'].numpy()
        
        skill_df = pd.DataFrame(skill_data, index=target_names)
        skill_overall = skill_df.mean(axis=0)
        skill_df.loc['OVERALL AVERAGE'] = skill_overall
        
        # Format skill score table
        rows = []
        headers = ['Target Variable'] + list(skill_df.columns)
        rows.append(headers)
        rows.append(['---'] * len(headers))
        
        for target in skill_df.index:
            row = [target]
            values = skill_df.loc[target]
            best_value = values.max()
            
            for model_name in skill_df.columns:
                value = skill_df.loc[target, model_name]
                formatted_value = f"{value:.4f}"
                
                if target != 'OVERALL AVERAGE' and abs(value - best_value) < 1e-6:
                    row.append(f"**{formatted_value}**")
                else:
                    row.append(formatted_value)
            
            rows.append(row)
        
        output_text += tabulate(rows, tablefmt="pipe", headers="firstrow") + "\n"
    
    # Add interpretation guidelines
    output_text += """

## Performance Guidelines

### Excellent Performance
- **R² > 0.8**: Captures most atmospheric physics variability
- **ACC > 0.9**: Excellent pattern recognition
- **NRMSE < 0.3**: Low error relative to natural variability

### Good Performance  
- **R² > 0.7**: Good physics capture
- **ACC > 0.8**: Strong pattern skills
- **NRMSE < 0.5**: Reasonable error levels

### Concerning Performance
- **R² < 0.5**: Limited physics understanding
- **ACC < 0.6**: Poor pattern recognition  
- **NRMSE > 0.8**: High error relative to variability

## Key Insights for Atmospheric Tendencies

1. **R² is most important** - Shows fraction of physics variance you've captured
2. **ACC crucial for tiny values** - Pattern skill matters more than absolute error
3. **NRMSE provides scale context** - Essential when dealing with different physical units
4. **Baseline independence** - These core metrics don't depend on training data baseline
5. **Perfect for subset training** - Works with your 10% sampling approach

## Adding Baseline Metrics

To add Skill Score and other baseline-dependent metrics:

1. **Calculate full training baseline separately**:
   ```python
   # From your full training data  
   train_target_mean = torch.mean(full_training_targets, dim=0)
   # Save to pickle file
   with open('full_training_baseline.pickle', 'wb') as f:
       pickle.dump(train_target_mean, f)
   ```

2. **Enable baseline metrics in this script**:
   ```python
   CALCULATE_BASELINE_METRICS = True
   FULL_TRAINING_BASELINE_PATH = 'path/to/full_training_baseline.pickle'
   ```

3. **Re-run this script** to get Skill Score and baseline comparisons

"""
    
    return output_text

# Generate the clean summary
output_text = create_clean_summary()

# Save the analysis
output_path = '/mydata/deepcloud/yves/final_results/3D_tendency_evaluation_baseline.md'
with open(output_path, 'w') as f:
    f.write(output_text)

print(f"\n=== CORE METRICS ANALYSIS COMPLETE ===")
print(f"Clean physics evaluation saved to: {output_path}")
print("\nFocus on these baseline-independent metrics:")
print("1. R² (primary) - fraction of atmospheric variance explained") 
print("2. ACC (critical) - pattern recognition for tiny tendencies")
print("3. NRMSE (essential) - error relative to natural variability")
print("4. NMAE (essential) - error relative to natural variability")
print("5. MAE (interpretable) - direct physical error in atmospheric units (scientific notation)")

if not CALCULATE_BASELINE_METRICS:
    print("\nTo add Skill Score and baseline metrics:")
    print("1. Calculate full training baseline separately and save to pickle")
    print("2. Set CALCULATE_BASELINE_METRICS = True")
    print("3. Set FULL_TRAINING_BASELINE_PATH to your baseline file")
    print("4. Re-run this script") 
else:
    print(f"\nBaseline metrics enabled using: {FULL_TRAINING_BASELINE_PATH}") 