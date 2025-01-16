import torch
import pickle
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
from scipy import stats


# Example: single model in your list
models = [
    {
        'name': 'AFNO-Emb128-concat-tendency-norm',
        'path': '/mydata/deepcloud/shared/results-temp/afno_column_1percent_Emb128_clean_tendency_normTarg/test'
    },
]

# We'll store:
#   - model_mae_heights:    per-height MAE for the model
#   - baseline_mae_heights: per-height MAE for the baseline
model_mae_heights = []
baseline_mae_heights = []

for mdl in models:
    test_path = mdl['path']
    print(f"Loading test files... ({test_path})")

    with open(join(test_path, 'y_true.pickle'), 'rb') as f:
        y_true = pickle.load(f)
    print(f'y_true shape: {y_true.shape}')  # Should be [batch, height, 7]

    with open(join(test_path, 'y_pred.pickle'), 'rb') as f:
        y_pred = pickle.load(f)
    print(f'y_pred shape: {y_pred.shape}')

    with open(join(test_path, 'train_target_mean.pickle'), 'rb') as f:
        train_target_mean = pickle.load(f)
    print(f'train_target_mean shape: {train_target_mean.shape}')  # [height, 7]

    # Since shape is [batch, height, features], average over dim=0
    dim_for_batch = 0

    # -------------------------------------------------
    # A) Model MAE vs. Height
    # -------------------------------------------------
    # result shape will become [height, 7]
    model_mae_h = torch.mean(torch.abs(y_true - y_pred), dim=dim_for_batch)
    #model_mae_h = torch.flip(model_mae_h, dims=[0])
    model_mae_heights.append(model_mae_h)

    # -------------------------------------------------
    # B) Baseline MAE vs. Height
    # -------------------------------------------------
    # (Predicting 'train_target_mean' for each sample)
    baseline_mae_h = torch.mean(torch.abs(y_true - train_target_mean), dim=dim_for_batch)
    #baseline_mae_h = torch.flip(baseline_mae_h, dims=[0])
    baseline_mae_heights.append(baseline_mae_h)

    # (Optional) Print overall MAE as scalar
    baseline_mae_scalar = baseline_mae_h.mean().item()
    model_mae_scalar    = model_mae_h.mean().item()
    ratio = model_mae_scalar / baseline_mae_scalar if baseline_mae_scalar != 0 else float('inf')
    improvement = (1 - ratio) * 100

    print(f'Baseline MAE (scalar): {baseline_mae_scalar:.15f}')
    print(f'Model MAE (scalar):    {model_mae_scalar:.15f}')
    print(f'MAE Ratio (Model / Baseline): {ratio:.15f}')
    print(f'Improvement (Model / Baseline): {improvement:.2f}%')

# -------------------------------------------------
# STEP 2)  Plot: per-feature lines vs. height
# -------------------------------------------------

# 7 output channels, with units for labeling:
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

height_size = model_mae_heights[0].shape[0]  # e.g. 70
height_range = range(height_size)            # or actual altitude if available

fig = plt.figure(figsize=(12, 18))

def add_subplot_mae(
    fig,
    height_vals,
    baseline_lines,  # list of Tensors, shape [height]
    model_lines,     # list of Tensors, shape [height]
    subplot_pos,
    models_names,
    title=None,
    ylabel=None,
    xlabel=None,
    log_scale=False
):
    """
    Plots baseline vs. model lines (MAE) for each model on the same subplot.
    """
    ax = fig.add_subplot(*subplot_pos)
    for b_line, m_line, name in zip(baseline_lines, model_lines, models_names):
        ax.plot(b_line, height_vals, 'r--', label=f"{name} Baseline MAE")
        ax.plot(m_line, height_vals, 'b-', label=f"{name} Model MAE")

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

# We'll create subplots in a 3x3 grid, one per feature (7 total)
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

# For a single legend, combine from the first subplot (or create a global legend):
ax_list[0].legend(fontsize=9, loc='upper right')

plt.tight_layout()
plt.savefig('/mydata/deepcloud/shared/results-temp/tendency-data-mae-baseline-vs-model.png',
            bbox_inches='tight', dpi=300)   
plt.show()




# Example: Calculate 95% confidence interval for MAE
def confidence_interval(data, confidence=0.95):
    mean = np.mean(data)
    sem = stats.sem(data)
    interval = stats.t.interval(confidence, len(data)-1, loc=mean, scale=sem)
    return interval

# Assuming you have per-sample MAEs
baseline_mae_samples = torch.abs(y_true - train_target_mean).mean(dim=0).flatten().numpy()
model_mae_samples = torch.abs(y_true - y_pred).mean(dim=0).flatten().numpy()

baseline_ci = confidence_interval(baseline_mae_samples)
model_ci = confidence_interval(model_mae_samples)

print(f'Baseline MAE 95% CI: {baseline_ci}')
print(f'Model MAE 95% CI: {model_ci}')
