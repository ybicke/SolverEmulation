import torch
import pickle
import argparse
from os.path import join, isfile
from matplotlib import pyplot as plt

# Add argument parsing to allow for command-line model path specification
parser = argparse.ArgumentParser(description='Visualize and evaluate tendency model predictions')
parser.add_argument('--model_path', type=str, default=None, help='Path to model test results directory')
args = parser.parse_args()

# Define models to evaluate
models = [
    {
        'name': 'AFNO',
        #'path': '/mydata/deepcloud/yves/results-temp/gnn_32_l2_tendency_normTarg/test'
        #'path': '/mydata/deepcloud/yves/results-temp/gnn_64_l3_tendency_normTarg/test'
        #'path': '/mydata/deepcloud/yves/results-temp/gnn3d_id39_tendency_32_l2_indep/test',
        'path': '/mydata/deepcloud/yves/results-temp/gnn3d_id39_tendency_32_l2/test',
        # 'path': '/mydata/deepcloud/yves/results-temp/gnn_128_l2_tendency_normTarg/test'

    },
]

# Override model path if provided via command line
if args.model_path:
    models = [{'name': 'Model', 'path': args.model_path}]

# We'll store:
#   - model_mae_heights:    per-height MAE for the model
#   - baseline_mae_heights: per-height MAE for the baseline
model_mae_heights = []
baseline_mae_heights = []

def save_metrics_to_file(metrics, file_path):
    """
    Save metrics dictionary to a formatted text file.
    """
    with open(file_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"TENDENCY MODEL EVALUATION METRICS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"MODEL: {metrics['model_name']}\n")
        f.write("-" * 50 + "\n\n")
        
        f.write("BASIC ERROR METRICS:\n")
        f.write(f"  Baseline MSE         : {metrics['baseline_mse']:.8e}\n")
        f.write(f"  Model MSE            : {metrics['model_mse']:.8e}\n")
        f.write(f"  MSE Ratio (M/B)      : {metrics['mse_ratio']:.4f}\n")
        f.write(f"  Improvement          : {metrics['improvement_pct']:.2f}%\n")
        f.write(f"  Baseline MAE         : {metrics['baseline_mae']:.8e}\n")
        f.write(f"  Model MAE            : {metrics['model_mae']:.8e}\n")
        f.write(f"  Baseline RMSE        : {metrics['rmse_baseline']:.8e}\n")
        f.write(f"  Model RMSE           : {metrics['rmse_model']:.8e}\n\n")
        
        f.write("SCALE-AWARE METRICS:\n")
        f.write(f"  Test set variance    : {metrics['var_true']:.8e}\n")
        f.write(f"  NRMSE                : {metrics['nrmse']:.4f}\n")
        f.write(f"  R² (explained var.)  : {metrics['r2']:.4f}\n")
        f.write(f"  ACC                  : {metrics['acc']:.4f}\n")
        f.write(f"  Skill vs. baseline   : {metrics['skill_score']:.4f}\n\n")
        
        f.write("INTERPRETATION GUIDE:\n")
        f.write("  - NRMSE < 1.0 indicates the model outperforms predicting the mean\n")
        f.write("  - R² closer to 1.0 indicates better prediction of variance\n")
        f.write("  - ACC closer to 1.0 indicates better prediction of patterns\n")
        f.write("  - Skill score ranges from -∞ to 1.0, with values > 0 indicating\n")
        f.write("    improvement over baseline and 1.0 being perfect prediction\n")
        
    print(f"Saved evaluation metrics to {file_path}")

for mdl in models:
    test_path = mdl['path']
    print(f"Loading test files... ({test_path})")

    with open(join(test_path, 'y_true.pickle'), 'rb') as f:
        y_true = pickle.load(f)
    print(f'y_true shape: {y_true.shape}')

    with open(join(test_path, 'y_pred.pickle'), 'rb') as f:
        y_pred = pickle.load(f)
    print(f'y_pred shape: {y_pred.shape}')

    with open(join(test_path, 'train_target_mean.pickle'), 'rb') as f:
        train_target_mean = pickle.load(f)
    print(f'train_target_mean shape: {train_target_mean.shape}')
    

    # -------------------------------------------------------------
    # COMPUTE COMPREHENSIVE EVALUATION METRICS 
    # -------------------------------------------------------------
    
    # 1. Basic error metrics
    model_mse = torch.mean((y_true - y_pred)**2)
    model_mae = torch.mean(torch.abs(y_true - y_pred))
    baseline_mse = torch.mean((y_true - train_target_mean)**2)
    baseline_mae = torch.mean(torch.abs(y_true - train_target_mean))
    
    rmse_model = torch.sqrt(model_mse)
    rmse_baseline = torch.sqrt(baseline_mse)
    
    mse_ratio = model_mse / baseline_mse
    improvement_pct = (1 - mse_ratio) * 100
    
    print(f"\n{'-'*50}")
    print(f"MODEL: {mdl['name']}")
    print(f"{'-'*50}")
    print(f'Baseline MSE: {baseline_mse:.8f}')
    print(f'Model MSE: {model_mse:.8f}')
    print(f'MSE Ratio (Model / Baseline): {mse_ratio:.4f}')
    print(f'Improvement: {improvement_pct:.2f}%')
    
    # 2. Scale-aware metrics using test set statistics
    def _flatten(t):
        """Flatten all but the last dimension so metrics treat every level equally."""
        return t.view(-1)
    
    # Flatten arrays for aggregate statistics
    y_true_flat = _flatten(y_true)
    y_pred_flat = _flatten(y_pred)

    # Calculate test set variance for proper evaluation metrics
    var_true = torch.var(y_true_flat, unbiased=False)
    std_true = torch.sqrt(var_true)
    print(f"Test set variance: {var_true:.8f} (used for metrics normalization)")

    # NRMSE - using test set standard deviation
    nrmse = rmse_model / std_true

    # R² (explained variance) - using test set variance
    r2 = 1 - model_mse / var_true

    # Anomaly Correlation Coefficient (ACC)
    y_true_anom = y_true_flat - torch.mean(y_true_flat)
    y_pred_anom = y_pred_flat - torch.mean(y_pred_flat)
    acc_num = torch.sum(y_true_anom * y_pred_anom)
    acc_den = torch.sqrt(torch.sum(y_true_anom ** 2) * torch.sum(y_pred_anom ** 2))
    acc = acc_num / acc_den

    # Skill score relative to baseline (climatology)
    skill_score = 1 - rmse_model / rmse_baseline

    print("\nScale‑aware metrics (aggregated over all outputs):")
    print(f"  NRMSE                : {nrmse:.4f}")
    print(f"  R² (explained var.)  : {r2:.4f}")
    print(f"  ACC                  : {acc:.4f}")
    print(f"  Skill vs. baseline   : {skill_score:.4f}\n")
    
    # Collect all metrics in a dictionary
    metrics = {
        'model_name': mdl['name'],
        'baseline_mse': baseline_mse.item(),
        'model_mse': model_mse.item(),
        'baseline_mae': baseline_mae.item(),
        'model_mae': model_mae.item(),
        'rmse_baseline': rmse_baseline.item(),
        'rmse_model': rmse_model.item(),
        'mse_ratio': mse_ratio.item(),
        'improvement_pct': improvement_pct.item(),
        'var_true': var_true.item(),
        'nrmse': nrmse.item(),
        'r2': r2.item(),
        'acc': acc.item(),
        'skill_score': skill_score.item()
    }
    
    # Save metrics to file
    metrics_file = join(test_path, 'evaluation_metrics.txt')
    save_metrics_to_file(metrics, metrics_file)
    
    
    
    
    
    # 3. Height-dependent metrics
    # Since shape is [batch, height, features], we average over dim=0
    dim_for_batch = 0

    # A) Model MAE vs. Height (and feature) - shape will be [height, 7]
    y_mae_h = torch.mean(torch.abs(y_true - y_pred), dim=dim_for_batch)
    model_mae_heights.append(y_mae_h)
    
    # B) Baseline MAE vs. Height, (Predicting 'train_target_mean' for each sample)
    baseline_mae_h = torch.mean(torch.abs(y_true - train_target_mean), dim=dim_for_batch)
    baseline_mae_heights.append(baseline_mae_h)

# -------------------------------------------------------------
# VISUALIZATION SECTION
# -------------------------------------------------------------

target_units = {
    "Sum of Temperature Tendency": "K s-1", 
    "Dynamical Temperature Tendency": "K s-1",
    "Sum of Zonal Wind Tendency": "m s-2",
    "Sum of Meridional Wind Tendency": "m s-2",
    "Convective Tend. Absolute Humidity": "kg m-3 s-1",
    "Convective Tend. Cloud Water Mass Density": "kg m-3 s-1",
    "Convective Tend. Cloud Ice Mass Density": "kg m-3 s-1"
}


models_name = [m['name'] for m in models]

height_size = model_mae_heights[0].shape[0]
height_range = range(height_size)  # or actual altitude if available

fig = plt.figure(figsize=(12, 18))

def add_subplot_mae(
    fig,
    height_vals,
    baseline_lines,   # list of Tensors, shape [height]
    model_lines,      # list of Tensors, shape [height]
    subplot_pos,
    models_names,
    title=None,
    ylabel=None,
    xlabel=None,
    log_scale=False
):
    """
    Plots baseline vs. model lines for each model on the same subplot.
    """
    ax = fig.add_subplot(*subplot_pos)
    for b_line, m_line, name in zip(baseline_lines, model_lines, models_names):
        ax.plot(b_line, height_vals, 'r--', label=f"{name} Baseline MAE")
        ax.plot(m_line, height_vals, 'b-', label=f"{name} Model MAE")

    if log_scale:
        ax.set_xscale('log')

    ax.grid(True)
    ax.invert_yaxis()
    # Increase tick label size
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    # Increase the font size of the scientific notation
    ax.xaxis.get_offset_text().set_fontsize(12)
    
    if title:
        ax.set_title(title if title else '', fontsize=13.5)  # Increase title size
    if ylabel:
        ax.set_ylabel(ylabel if ylabel else '', fontsize=14)  # Increase label size
    if xlabel:
        ax.set_xlabel(xlabel if xlabel else '', fontsize=14)  # Increase label size

    return ax

# Create subplots in a 3x3 grid, one per feature (7 total)
ax_list = []
for i, (feat_label, units) in enumerate(target_units.items()):
    subplot_idx = (3, 3, i+1)

    # For each model, slice the i-th feature from [height, 7]
    baseline_feat_lines = [bm[:, i] for bm in baseline_mae_heights]
    model_feat_lines    = [mm[:, i] for mm in model_mae_heights]

    ax = add_subplot_mae(
        fig=fig,
        height_vals=height_range,
        baseline_lines=baseline_feat_lines,
        model_lines=model_feat_lines,
        subplot_pos=subplot_idx,
        models_names=models_name,
        title=feat_label,
        ylabel='Height index' if i in [0, 3, 6] else None,
        xlabel=f"MAE [{units}]",
        log_scale=False  # or False for linear scale
    )
    ax_list.append(ax)

# Legend on the first subplot
ax_list[0].legend(fontsize=12, loc='best')

plt.tight_layout()

# Save to either specified model path or default location
output_path = join(test_path, 'tendency-mae-baseline-vs-model.png') if len(models) == 1 else '/mydata/deepcloud/shared/results-temp/tendency-data-mae-baseline-vs-model.png'
plt.savefig(output_path, bbox_inches='tight', dpi=300)   
print(f"Saved visualization to {output_path}")

plt.show()