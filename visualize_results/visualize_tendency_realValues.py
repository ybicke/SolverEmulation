import torch
import pickle
import numpy as np
from os.path import join
from matplotlib import pyplot as plt

# Define your models
models = [
    {'name': 'AFNO','path': '/mydata/deepcloud/yves/results-temp/afno_column_1percent_Emb128_clean_tendency_normTarg/test'},
    # Add more models here if needed
]

# Step 1: Load Data and Compute Per-Height Means
for mdl in models:
    test_path = mdl['path']
    print(f"Loading test files... ({test_path})")

    # Load y_true
    with open(join(test_path, 'y_true.pickle'), 'rb') as f:
        y_true = pickle.load(f)
    print(f'y_true shape: {y_true.shape}')  

    # Load y_pred
    with open(join(test_path, 'y_pred.pickle'), 'rb') as f:
        y_pred = pickle.load(f)
    print(f'y_pred shape: {y_pred.shape}')  

    # Load train_target_mean (not needed for this plot, but kept for reference)
    with open(join(test_path, 'train_target_mean.pickle'), 'rb') as f:
        train_target_mean = pickle.load(f)
    print(f'train_target_mean shape: {train_target_mean.shape}')
    
    # Convert to tensors (using clone() to avoid warning)
    if isinstance(y_true, torch.Tensor):
        y_true_tensor = y_true.clone().detach()
    else:
        y_true_tensor = torch.tensor(y_true)
        
    if isinstance(y_pred, torch.Tensor):
        y_pred_tensor = y_pred.clone().detach()
    else:
        y_pred_tensor = torch.tensor(y_pred)
    
    # Check and reshape if necessary
    if y_true_tensor.shape[-1] == 490:  # Flattened shape detected
        y_true_tensor = y_true_tensor.reshape(-1, 70, 7)
        y_pred_tensor = y_pred_tensor.reshape(-1, 70, 7)
        print(f'Reshaped y_true: {y_true_tensor.shape}')
        print(f'Reshaped y_pred: {y_pred_tensor.shape}')

    # Select a subset of samples
    num_samples = 1  # Choose the number of samples to plot
    sample_indices = np.random.choice(y_true_tensor.shape[0], num_samples, replace=False)
    y_true_subset = y_true_tensor[sample_indices]  # [num_samples, 70, 7]
    y_pred_subset = y_pred_tensor[sample_indices]  # [num_samples, 70, 7]

    # Compute the mean over the samples
    y_true_subset_mean = torch.mean(y_true_subset, dim=0)  # [70, 7]
    y_pred_subset_mean = torch.mean(y_pred_subset, dim=0)  # [70, 7]

# Define target units for plotting
target_units = {
    "Sum of Temperature Tendency": "K s$^{-1}$", 
    "Dynamical Temperature Tendency": "K s$^{-1}$",
    "Sum of Zonal Wind Tendency": "m s$^{-2}$",
    "Sum of Meridional Wind Tendency": "m s$^{-2}$",
    "Convective Tend. Absolute Humidity": "kg m$^{-3}$ s$^{-1}$",
    "Convective Tend. Cloud Water Mass Density": "kg m$^{-3}$ s$^{-1}$",
    "Convective Tend. Cloud Ice Mass Density": "kg m$^{-3}$ s$^{-1}$"
}

# Extract model names for labeling
models_name = [m['name'] for m in models]

# Define the height range based on the number of height levels
height_size = y_true_subset_mean.shape[0]  # e.g., 70
height_range = list(range(height_size))    # 0 to 69

# Initialize the figure
fig = plt.figure(figsize=(12, 18))

# Define the plotting function
def add_subplot_prediction(
    fig,
    height_vals,
    true_values,     # Tensor shape [70]
    pred_values,     # Tensor shape [70]
    subplot_pos,
    model_name,
    title=None,
    ylabel=None,
    xlabel=None,
    log_scale=False
):
    ax = fig.add_subplot(*subplot_pos)
    ax.plot(true_values.numpy(), height_vals, 'r--', label=f"{model_name} True Mean")
    ax.plot(pred_values.numpy(), height_vals, 'b-', label=f"{model_name} Predicted Mean")

    if log_scale:
        ax.set_xscale('log')

    ax.grid(True)
    ax.invert_yaxis()
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.xaxis.get_offset_text().set_fontsize(12)
    
    if title:
        ax.set_title(title, fontsize=13.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=14)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=14)

    return ax

# Create subplots for each feature
ax_list = []
for i, (feat_label, units) in enumerate(target_units.items()):
    subplot_idx = (3, 3, i+1)  # 3x3 grid

    # Extract the specific feature column for all heights
    true_feat = y_true_subset_mean[:, i]  # shape [70]
    pred_feat = y_pred_subset_mean[:, i]  # shape [70]

    ax = add_subplot_prediction(
        fig=fig,
        height_vals=height_range,
        true_values=true_feat,
        pred_values=pred_feat,
        subplot_pos=subplot_idx,
        model_name=models_name[0],  # Using the first (and only) model
        title=feat_label,
        ylabel='Height Index' if i in [0, 3, 6] else None,
        xlabel=f"[{units}]",
        log_scale=False
    )
    ax_list.append(ax)

# Add a legend to the first subplot
ax_list[0].legend(fontsize=12, loc='best')

# Adjust layout and save the figure
plt.tight_layout(rect=[0, 0, 0.90, 1])
plt.savefig('/mydata/deepcloud/yves/results-temp/tendency-data-prediction-true-subset.png',
            bbox_inches='tight', dpi=300)
plt.show()