import torch
import pickle
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter

# -----------------------------------------------------------------------------
#   Libraries & global matplotlib setup
# -----------------------------------------------------------------------------

# Use a clean scientific style
plt.style.use('seaborn-v0_8-whitegrid')  # requires Matplotlib >=3.7; fallback ok

# Increase default font sizes a bit
plt.rcParams.update({
    'axes.titlesize': 13,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

# Configuration options
# Choose whether to use all samples or only selected ones
use_all_samples = True  # Set to True to use all samples from the test set

# If not using all samples, specify which samples to use
selected_samples = [112]  # List of sample indices to visualize

# Set to True to save to the model's test directory
save_to_test_path = True
# Optional additional save path
# additional_save_path = '/mydata/deepcloud/yves/Tendency-normTarget-true_vs_pred-sample.png'

# Define models to evaluate
models = [
    {
        #'path': '/mydata/deepcloud/yves/results-temp/afno_column_1percent_Emb128_clean_tendency_normTarg/test'
        #'path': '/mydata/deepcloud/yves/results-temp/gnn_32_l2_tendency_normTarg/test'
        #'path': '/mydata/deepcloud/yves/results-temp/gnn_64_l3_tendency_normTarg/test'
        #'path': '/mydata/deepcloud/yves/results-temp/gnn_128_l2_tendency_normTarg/test'
        


    },
    # Add more models as needed for comparison
    # {'name': 'Model2', 'path': '/path/to/model2/test'},
]

# Containers for plotting
y_pred_hs = []   # will hold one (height, channels) tensor per model
y_true_h = None  # will store the ground‑truth tensor (height, channels) once

for model in models:
    test_path = model['path']
    print(f'Loading test files... ({test_path})')

    with open(join(test_path, 'y_true.pickle'), 'rb') as handle:
        y_true = pickle.load(handle)
    print(f'y_true shape: {y_true.shape}')

    with open(join(test_path, 'y_pred.pickle'), 'rb') as handle:
        y_pred = pickle.load(handle)
    print(f'y_pred shape: {y_pred.shape}')

    # Decide whether to use all samples or only selected ones
    if not use_all_samples:
        # Use only selected samples
        y_true = y_true[selected_samples]
        y_pred = y_pred[selected_samples]
        print(f'Using selected samples: {selected_samples}')
    else:
        print(f'Using all {len(y_true)} samples')

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

    # ------------------------------------------------------------------
    # Collapse the (optional) sample dimension so that we have one curve
    # per height level. If several indices are given we take the mean over
    # those selected samples; if only one index is given we simply squeeze
    # that dimension.
    # ------------------------------------------------------------------
    if y_true.ndim == 3 and y_true.shape[0] > 1:
        y_true_h_tmp = torch.mean(y_true, dim=0)  # (H, C)
        y_pred_h_tmp = torch.mean(y_pred, dim=0)
    else:
        y_true_h_tmp = y_true.squeeze(0)  # (H, C)
        y_pred_h_tmp = y_pred.squeeze(0)

    # store predictions; ground truth needs to be stored only once
    if y_true_h is None:
        y_true_h = y_true_h_tmp
    y_pred_hs.append(y_pred_h_tmp)

    # ------------------------------------------------------------------
    # MAE calculation kept for reference (might still be useful) – comment
    # out if you no longer need it.
    # y_mae_h = torch.mean(torch.abs(y_true - y_pred), dim=0)
    # y_mae_hs.append(y_mae_h)
    
   

target_units = {
    "Sum of Temperature Tendency": "K s-1", 
    "Dynamical Temperature Tendency": "K s-1",
    "Sum of Zonal Wind Tendency": "m s-2",
    "Sum of Meridional Wind Tendency": "m s-2",
    "Convective Tend. Absolute Humidity": "kg m-3 s-1",
    "Convective Tend. Cloud Water Mass Density": "kg m-3 s-1",
    "Convective Tend. Cloud Ice Mass Density": "kg m-3 s-1"
}

# ------------------------------------------------------------------
#                      P L O T    T R U T H   &   P R E D
# ------------------------------------------------------------------

# Prepare the figure
models_name = [model['name'] for model in models]

# Define colors for consistent plotting - true is black, predictions use a consistent color set
colors = ['black']  # True value is always black
pred_colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink']  # For predictions

# Height indices (0 ... H-1)
height_range = range(y_true_h.shape[0])

fig = plt.figure(figsize=(12, 18))

def add_subplot_true_pred(
    fig,
    height_vals,
    true_data,       # Tensor, shape [height, channel]
    pred_data_list,  # list of Tensors, shape [height, channel]
    subplot_pos,
    models_names,
    channel_idx,
    title=None,
    ylabel=None,
    xlabel=None
):
    """
    Plots true vs predicted lines for each model on the same subplot.
    """
    ax = fig.add_subplot(*subplot_pos)
    
    # Plot TRUE (black solid)
    ax.plot(
        true_data[:, channel_idx],
        height_vals,
        label='True' if channel_idx == 0 else None,
        color='black',
        linewidth=2,
    )

    # Plot PREDICTIONS for each model
    for i, (pred_data, model_name) in enumerate(zip(pred_data_list, models_names)):
        ax.plot(
            pred_data[:, channel_idx],
            height_vals,
            label=f'Pred • {model_name}' if channel_idx == 0 else None,
            linestyle='--',
            color=pred_colors[i % len(pred_colors)]  # Use consistent colors from list
        )

    ax.grid(True)
    ax.invert_yaxis()
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    if title:
        ax.set_title(title, fontsize=13.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=14)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=14)
    
    # Setup scientific notation with proper formatting
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.xaxis.set_major_formatter(formatter)
    
    # Set appropriate number of ticks
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    
    # Calculate appropriate axis limits
    data_values = [true_data[:, channel_idx]]
    for pred in pred_data_list:
        data_values.append(pred[:, channel_idx])
    
    min_val = min(d.min().item() for d in data_values)
    max_val = max(d.max().item() for d in data_values)
    
    # Add 5% padding to the range
    range_val = max_val - min_val
    min_val = min_val - 0.05 * range_val
    max_val = max_val + 0.05 * range_val
    
    # Set the limits
    ax.set_xlim(min_val, max_val)
    
    return ax

# Create one subplot for each of the 7 target channels in a 3 × 3 grid
ax_list = []

for i, (label, units) in enumerate(target_units.items()):
    subplot_idx = (3, 3, i + 1)

    ax = add_subplot_true_pred(
        fig=fig,
        height_vals=height_range,
        true_data=y_true_h,
        pred_data_list=y_pred_hs,
        subplot_pos=subplot_idx,
        models_names=models_name,
        channel_idx=i,
        title=label,
        ylabel='Height index' if i in [0, 3, 6] else None,
        xlabel=f'[$\\mathrm{{{units}}}$]'
    )
    
    ax_list.append(ax)

# ------------------------------ Legend & Layout ------------------------------

# Create a single legend at the bottom centered across the figure with clear labels
handles, labels = [], []
for ax in ax_list:
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in labels:
            handles.append(h)
            labels.append(l)

# Place legend below all subplots
fig.legend(handles, labels, loc='lower center', ncol=len(labels),
           bbox_to_anchor=(0.5, 0.0), frameon=False)

# Tight layout with extra space at bottom for legend
fig.tight_layout(rect=[0, 0.05, 1, 1])

# Save the figure
# Create a descriptive sample suffix for filenames
if use_all_samples:
    sample_suffix = "all_samples"
else:
    sample_suffix = f"samples_{'-'.join(map(str, selected_samples))}" if len(selected_samples) > 1 else f"sample_{selected_samples[0]}"

# Save to each model's test path if requested
if save_to_test_path:
    for model in models:
        output_path = join(model['path'], f'tendency-true-vs-pred-{sample_suffix}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Saved visualization to {output_path}")

# Save to additional path if provided
# if additional_save_path:
#     plt.savefig(additional_save_path, bbox_inches='tight', dpi=300)
#     print(f"Saved visualization to {additional_save_path}")

plt.show()


