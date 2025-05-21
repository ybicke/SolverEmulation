import torch
import pickle
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Use a clean scientific style
plt.style.use('seaborn-v0_8-whitegrid')

# Increase default font sizes
plt.rcParams.update({
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

# Models to compare
models = [
    {
        'name': 'AFNO',
        'path': '/mydata/deepcloud/yves/results-temp/afno_column_1percent_Emb128_clean_tendency_normTarg/test'
    },
    #{
    #    'name': 'GNN-32-L2',
    #    'path': '/mydata/deepcloud/yves/results-temp/gnn_32_l2_tendency_normTarg/test'
    #},
    #{
    #    'name': 'GNN-64-L3',
    #    'path': '/mydata/deepcloud/yves/results-temp/gnn_64_l3_tendency_normTarg/test'
    #},
    {
        'name': 'GNN-128-L2',
        'path': '/mydata/deepcloud/yves/results-temp/gnn_128_l2_tendency_normTarg/test'
    }
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

# Choose whether to use all samples or selected samples
use_all_samples = True
selected_samples = [0]  # Change this to select specific samples

# Store all model data
model_names = [model['name'] for model in models]
y_pred_hs = []  # Will store predictions from each model
y_true_h = None  # Will store ground truth (only need one copy)

# Load and process data for each model
for model_idx, model in enumerate(models):
    test_path = model['path']
    print(f'Loading test files... ({test_path})')

    with open(join(test_path, 'y_true.pickle'), 'rb') as handle:
        y_true = pickle.load(handle)
    
    with open(join(test_path, 'y_pred.pickle'), 'rb') as handle:
        y_pred = pickle.load(handle)
    
    # Decide whether to use all samples or only selected ones
    if not use_all_samples:
        y_true = y_true[selected_samples]
        y_pred = y_pred[selected_samples]
        print(f'Using selected samples: {selected_samples}')
    else:
        print(f'Using all {len(y_true)} samples')
    
    # Convert to PyTorch tensors if needed
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred)
    
    # Reshape if needed
    if y_true.shape[-1] == 490:
        y_true = y_true.reshape(-1, 70, 7)
        y_pred = y_pred.reshape(-1, 70, 7)
    
    # Store the ground truth from the first model only
    if model_idx == 0:
        # If multiple samples, take the mean; otherwise squeeze the batch dimension
        if y_true.ndim == 3 and y_true.shape[0] > 1:
            y_true_h = torch.mean(y_true, dim=0)  # (H, C)
        else:
            y_true_h = y_true.squeeze(0)  # (H, C)
    
    # For predictions, always store
    if y_pred.ndim == 3 and y_pred.shape[0] > 1:
        y_pred_h = torch.mean(y_pred, dim=0)  # (H, C)
    else:
        y_pred_h = y_pred.squeeze(0)  # (H, C)
    
    y_pred_hs.append(y_pred_h)

# Get height range
height_range = range(y_true_h.shape[0])

# Define colors and line styles for models
true_color = 'black'
pred_colors = ['blue', 'red', 'green', 'purple']
line_styles = ['-', '--', '-.', ':']

# Create the consolidated plot with separate subplots
fig, axes = plt.subplots(1, 7, figsize=(20, 8), sharey=True)
fig.subplots_adjust(wspace=0.3)

for i, (label, unit) in enumerate(target_units.items()):
    ax = axes[i]
    
    # Plot true values first
    ax.plot(
        y_true_h[:, i],
        height_range,
        label='True' if i == 0 else None,
        color=true_color,
        linewidth=2.5
    )
    
    # Plot each model
    for j, (model_name, y_pred) in enumerate(zip(model_names, y_pred_hs)):
        ax.plot(
            y_pred[:, i], 
            height_range,
            label=model_name if i == 0 else None,
            color=pred_colors[j % len(pred_colors)],
            linestyle=line_styles[j % len(line_styles)],
            linewidth=1.5
        )
    
    # Format subplot
    ax.set_title(label.replace(" ", "\n"), fontsize=11)
    ax.grid(True, alpha=0.7)
    ax.invert_yaxis()
    
    # Add x-label
    ax.set_xlabel(f'[{unit}]', fontsize=12)
    
    # Add y-label only to the first subplot
    if i == 0:
        ax.set_ylabel('Height Level', fontsize=14)
    
    # Setup scientific notation with proper formatting
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.xaxis.set_major_formatter(formatter)
    
    # Set appropriate x-limits for all data
    all_values = [y_true_h[:, i]] + [y_pred[:, i] for y_pred in y_pred_hs]
    min_val = min(v.min().item() for v in all_values)
    max_val = max(v.max().item() for v in all_values)
    range_val = max_val - min_val
    padding = range_val * 0.1
    ax.set_xlim(min_val - padding, max_val + padding)

# Add a single legend for all subplots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, 
    labels, 
    loc='upper center', 
    bbox_to_anchor=(0.5, 0.02),
    ncol=len(models) + 1,  # +1 for the true values
    fontsize=14
)

# Add a super title
sample_desc = "All Samples (Mean)" if use_all_samples else f"Sample(s) {', '.join(map(str, selected_samples))}"
fig.suptitle(f'True vs. Predicted Values Across All Models - {sample_desc}', fontsize=16, y=0.98)

# Adjust layout
plt.tight_layout(rect=[0, 0.06, 1, 0.95])

# Save the figure
output_path = '/mydata/deepcloud/yves/results-temp/true-vs-pred-all-models.png'
plt.savefig(output_path, bbox_inches='tight', dpi=300)

print(f"Visualization saved to: {output_path}")
plt.show() 