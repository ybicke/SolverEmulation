import torch
import pickle
import numpy as np
from os.path import join
from matplotlib import pyplot as plt

# Define your models
models = [
    {'name': 'AFNO','path': '/mydata/deepcloud/shared/results-temp/afno_column_1percent_Emb128_clean_tendency_normTarg/test'},
    
    # {
    #     'name': 'RF-concat-tendency-norm',
    #     'path': '/mydata/deepcloud/shared/results-temp/afno_column_1percent_Emb128_clean_tendency_RF/test'
    # },
    
    
    # Add more models here if needed
]

# Initialize lists to store per-height mean predictions and true values
model_pred_means = []
model_true_means = []

# Step 1: Load Data and Compute Per-Height Means
for mdl in models:
    test_path = mdl['path']
    print(f"Loading test files... ({test_path})")

    # Load y_true
    with open(join(test_path, 'y_true.pickle'), 'rb') as f:
        y_true = pickle.load(f)
    print(f'y_true shape: {y_true.shape}')  # Expected: [540672, 70, 7]

    # Load y_pred
    with open(join(test_path, 'y_pred.pickle'), 'rb') as f:
        y_pred = pickle.load(f)
    print(f'y_pred shape: {y_pred.shape}')  # Expected: [540672, 70, 7]

    # Load train_target_mean (not needed for this plot, but kept for reference)
    with open(join(test_path, 'train_target_mean.pickle'), 'rb') as f:
        train_target_mean = pickle.load(f)
    print(f'train_target_mean shape: {train_target_mean.shape}')  # Expected: [70, 7]
    
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

    # Convert to tensors
    y_true_tensor = torch.tensor(y_true)   # [540672, 70, 7]
    y_pred_tensor = torch.tensor(y_pred)   # [540672, 70, 7]

    # Compute mean over the batch dimension (dim=0)
    # Resulting shape: [70, 7]
    y_true_mean = torch.mean(y_true_tensor, dim=0)
    y_pred_mean = torch.mean(y_pred_tensor, dim=0)

    # Optionally flip the vertical axis so index 0 is top
    #y_true_mean_flipped = torch.flip(y_true_mean, dims=[0])   # [70, 7]
    #y_pred_mean_flipped = torch.flip(y_pred_mean, dims=[0])   # [70, 7]

    # Append to lists
    model_true_means.append(y_true_mean)
    model_pred_means.append(y_pred_mean)


# Step 2: Define Target Units for Plotting
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
height_size = model_true_means[0].shape[0]  # e.g., 70
height_range = range(height_size)            # 0 to 69

# Initialize the figure
fig = plt.figure(figsize=(12, 18))

# Define the plotting function
def add_subplot_prediction(
    fig,
    height_vals,
    true_lines,      # list of Tensors, shape [height]
    pred_lines,      # list of Tensors, shape [height]
    subplot_pos,
    models_names,
    title=None,
    ylabel=None,
    xlabel=None,
    log_scale=False  # Set to False for linear scale
):
    """
    Plots true vs. model predicted lines for each model on the same subplot.
    
    Args:
        fig: The matplotlib figure object.
        height_vals: Iterable of height indices.
        true_lines: List of Tensors containing true means per height.
        pred_lines: List of Tensors containing model predicted means per height.
        subplot_pos: Tuple indicating subplot grid position (e.g., (3, 3, 1)).
        models_names: List of model names for labeling.
        title: Title of the subplot.
        ylabel: Y-axis label.
        xlabel: X-axis label.
        log_scale: Boolean indicating whether to use logarithmic scale on X-axis.
    """
    ax = fig.add_subplot(*subplot_pos)
    for true_line, pred_line, name in zip(true_lines, pred_lines, models_names):
        ax.plot(true_line.numpy(), height_vals, 'r--', label=f"{name} True Mean")
        ax.plot(pred_line.numpy(), height_vals, 'b-', label=f"{name} Predicted Mean")

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

# Step 3: Create Subplots for Each Feature
ax_list = []
for i, (feat_label, units) in enumerate(target_units.items()):
    subplot_idx = (3, 3, i+1)  # 3x3 grid

    # Extract the i-th feature across all heights for true and model
    true_feat_lines = [tm[:, i] for tm in model_true_means]    # List of [70]
    pred_feat_lines = [pm[:, i] for pm in model_pred_means]    # List of [70]

    ax = add_subplot_prediction(
        fig=fig,
        height_vals=height_range,
        true_lines=true_feat_lines,
        pred_lines=pred_feat_lines,
        subplot_pos=subplot_idx,
        models_names=models_name,
        title=feat_label,
        ylabel='Height Index' if i in [0, 3, 6] else None,
        xlabel=f"Mean [{units}]",
        log_scale=False  # Changed to False for linear scale
    )
    ax_list.append(ax)

# Step 4: Add a Single Legend to Avoid Clutter
# Extract handles and labels from the first subplot
handles, labels = ax_list[0].get_legend_handles_labels()
# Legend on the first subplot
ax_list[0].legend(fontsize=12, loc='best')

# Adjust layout and save the figure
plt.tight_layout(rect=[0, 0, 0.90, 1])  # Make space for the global legend
plt.savefig('/mydata/deepcloud/shared/results-temp/tendency-data-prediction-true-new.png',
            bbox_inches='tight', dpi=300)
plt.show()
 