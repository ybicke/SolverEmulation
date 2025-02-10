import torch
import pickle
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

models = [
        {
        'name': 'RF-concat-tendency-norm',
        'path': '/mydata/deepcloud/shared/results-temp/afno_column_1percent_Emb128_clean_tendency_RF/test'
    }, 
]

# We'll store:
#   - model_mae_heights:  per-height MAE for the model
#   - baseline_mse_heights: per-height MSE for the baseline
#   - model_mse_heights:    per-height MSE for the model
model_mae_heights = []
baseline_mse_heights = []
model_mse_heights = []

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
    
    # Convert y_true and y_pred to PyTorch tensors if they are numpy arrays
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred)
    
    # Check and reshape if necessary
    if y_true.shape[-1] == 490:  # Flattened shape detected
        y_true = y_true.reshape(-1, 70, 7)  # Reshape to [batch, height, features]
        y_pred = y_pred.reshape(-1, 70, 7)
        print(f'Reshaped y_true: {y_true.shape}')
        print(f'Reshaped y_pred: {y_pred.shape}')


    # Since shape is [batch, height, features], we average over dim=0
    dim_for_batch = 0

    # A) Model MAE vs. Height (and feature) result shape will become [height, 7]
    y_mae_h = torch.mean(torch.abs(y_true - y_pred), dim=dim_for_batch)
    #y_mae_h = torch.flip(y_mae_h, dims=[0])
    model_mae_heights.append(y_mae_h)
    
    # B) Baseline MSE vs. Height, (Predicting 'train_target_mean' for each sample)
    baseline_mse_h = torch.mean((y_true - train_target_mean) ** 2, dim=dim_for_batch)
    #baseline_mse_h = torch.flip(baseline_mse_h, dims=[0])
    baseline_mse_heights.append(baseline_mse_h)
    
    # C) Model MSE vs. Height
    model_mse_h = torch.mean((y_true - y_pred) ** 2, dim=dim_for_batch)
    #model_mse_h = torch.flip(model_mse_h, dims=[0])
    model_mse_heights.append(model_mse_h)

    # Print overall MSE scalars for reference
    baseline_mse_scalar = baseline_mse_h.mean().item()
    model_mse_scalar = model_mse_h.mean().item()
    ratio = model_mse_scalar / baseline_mse_scalar
    improvement = (1 - ratio) * 100

    print(f'Baseline MSE (scalar): {baseline_mse_scalar:.15f}')
    print(f'Model MSE (scalar):    {model_mse_scalar:.15f}')
    print(f'MSE Ratio (Model / Baseline): {ratio:.15f}')
    print(f'Improvement (Model / Baseline): {improvement:.2f}%')
    

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



def add_subplot_mse(
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
        ax.plot(b_line, height_vals, 'r--', label=f"{name} Baseline MSE")
        ax.plot(m_line, height_vals, 'b-', label=f"{name} Model MSE")

    if log_scale:
        ax.set_xscale('log')

    ax.grid(True)
    ax.invert_yaxis()
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)

    return ax

# Create subplots in a 3x3 grid, one per feature (7 total)
ax_list = []
for i, (feat_label, units) in enumerate(target_units.items()):
    subplot_idx = (3, 3, i+1)

    # For each model, slice the i-th feature from [height, 7]
    baseline_feat_lines = [bm[:, i] for bm in baseline_mse_heights]
    model_feat_lines    = [mm[:, i] for mm in model_mse_heights]

    ax = add_subplot_mse(
        fig=fig,
        height_vals=height_range,
        baseline_lines=baseline_feat_lines,
        model_lines=model_feat_lines,
        subplot_pos=subplot_idx,
        models_names=models_name,
        title=feat_label,
        ylabel='Height index' if i in [0, 3, 6] else None,
        xlabel=f"MSE [{units}]",
        log_scale=False  # or False for linear scale
    )
    ax_list.append(ax)

# Legend on the first subplot
ax_list[0].legend(fontsize=9, loc='upper right')

plt.tight_layout()
plt.savefig('/mydata/deepcloud/shared/results-temp/tendency-data-mse-baseline-vs-model-RF.png',
            bbox_inches='tight', dpi=300)   
plt.show()