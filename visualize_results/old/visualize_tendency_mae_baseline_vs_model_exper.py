import torch
import pickle
import numpy as np
import os
from os.path import join
from matplotlib import pyplot as plt
from scipy import stats
import matplotlib.cm as cm


# Example: single model in your list
models = [
    {
        'name': 'AFNO-Emb128-concat-tendency-norm',
        'path': '/mydata/deepcloud/shared/results-temp/afno_column_1percent_Emb128_clean_tendency_normTarg/test'
    },
        {
        'name': 'RF-concat-tendency-norm',
        'path': '/mydata/deepcloud/shared/results-temp/afno_column_1percent_Emb128_clean_tendency_RF/test'
    }, 
    
]


# Load the shared train target mean from a specific model path
mean_file_path = '/mydata/deepcloud/shared/results-temp/afno_column_1percent_Emb128_clean_tendency_normTarg/test/train_target_mean.pickle'

if os.path.exists(mean_file_path):
    with open(mean_file_path, 'rb') as f:
        train_target_mean = pickle.load(f)
    print(f"Loaded train target mean from: {mean_file_path}")
else:
    raise FileNotFoundError(f"Train target mean file not found: {mean_file_path}")


# We'll store:
#   - model_mae_heights:    per-height MAE for the model
#   - baseline_mae_heights: per-height MAE for the baseline
baseline_mae_heights = []
model_mae_heights = []



# Loop through models
for mdl in models:
    test_path = mdl['path']
    print(f"Loading test files... ({test_path})")

    with open(join(test_path, 'y_true.pickle'), 'rb') as f:
        y_true = pickle.load(f)
    print(f'y_true shape: {y_true.shape}')  # Should be [batch, height, 7]

    with open(join(test_path, 'y_pred.pickle'), 'rb') as f:
        y_pred = pickle.load(f)
    print(f'y_pred shape: {y_pred.shape}')

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

    # Compute the current baseline MAE
    current_baseline_mae_h = torch.mean(torch.abs(y_true - train_target_mean), dim=0)

    # Store the baseline MAE for each model
    baseline_mae_heights.append(current_baseline_mae_h)

    # A) Model MAE vs. Height
    #    shape -> [height, 7] after we average over batch dimension (dim=0)
    model_mae_h = torch.mean(torch.abs(y_true - y_pred), dim=0)
    model_mae_heights.append(model_mae_h)

    

    # Store the baseline MAE only once (or verify it's the same)
    if baseline_mae_heights is None:
        baseline_mae_heights = current_baseline_mae_h
    else:
        # You could verify that current_baseline_mae_h is the same if desired
        pass

    # Print overall MAE and improvement
    baseline_mae_scalar = current_baseline_mae_h.mean().item()
    model_mae_scalar = model_mae_h.mean().item()
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

def add_subplot_mae(fig, height_vals, baseline_lines, model_lines, subplot_pos,
                    models_names, title=None, ylabel=None, xlabel=None, log_scale=False):
    ax = fig.add_subplot(*subplot_pos)

    colors = plt.cm.viridis(np.linspace(0, 1, len(models_names)))

    for i, (baseline_line, model_line, model_name) in enumerate(zip(baseline_lines, model_lines, models_names)):
        ax.plot(baseline_line, height_vals, linestyle='--', color='red', label=f'Baseline ({model_name})')
        ax.plot(model_line, height_vals, linestyle='-', color=colors[i], label=f'Model ({model_name})')

    ax.invert_yaxis()
    ax.grid()

    if log_scale:
        ax.set_xscale('log')
        ax.minorticks_off()
    else:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

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

    # Slice i-th feature from [height, 7] for each model
    baseline_feat_lines = [bm[:, i] for bm in baseline_mae_heights]
    model_feat_lines    = [m[:, i] for m in model_mae_heights]

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
        log_scale=False  # True if you want a log scale
    )
    ax_list.append(ax)


# Create a single legend from the first subplot
ax_list[0].legend(fontsize=9, loc='upper right')

plt.tight_layout()
plt.savefig(
    '/mydata/deepcloud/shared/results-temp/tendency-mae-baseline-vs-model-afno-rf.png',
    bbox_inches='tight',
    dpi=300
)
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
