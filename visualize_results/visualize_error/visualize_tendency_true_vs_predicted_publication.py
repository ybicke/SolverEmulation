import torch
import pickle
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter
import os

# -----------------------------------------------------------------------------
#   Libraries & global matplotlib setup
# -----------------------------------------------------------------------------

# Use a clean scientific style
plt.style.use('seaborn-v0_8-whitegrid')  # requires Matplotlib >=3.7; fallback ok

# Increase default font sizes for publication
plt.rcParams.update({
    'axes.titlesize': 10,      # Reduced for publication
    'axes.labelsize': 9,       # Reduced for publication
    'xtick.labelsize': 8,      # Reduced for publication
    'ytick.labelsize': 8,      # Reduced for publication
    'legend.fontsize': 9,      # Reduced for publication
    'figure.dpi': 150,         # Higher DPI for publication
    'savefig.dpi': 300,        # High quality save
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'grid.alpha': 1,         # Lighter grid
    'grid.linewidth': 1,     # Thinner grid lines
    'axes.grid': True,         # Enable grid by default
    'axes.axisbelow': True,    # Grid behind plot elements
})

# Configuration options
# Choose whether to use all samples or only selected ones
use_all_samples = True  # Set to True to use all samples from the test set

# If not using all samples, specify which samples to use
selected_samples = [112]  # List of sample indices to visualize

# Set to True to save to the model's test directory
save_to_test_path = False  # Changed to False since we want to save to dedicated folder

# === PLOT CONFIGURATION ===
# Manually specify plot name and subfolder
PLOT_NAME = "3D_GT_vs_GNN_no_dropout"  # Change this for each plot scenario
SUBFOLDER = "true_vs_predicted"  # Options: "MAE", "true_vs_predicted", or any custom folder name

# Examples of plot names for different scenarios:
# "3D_vs_1D_GNN_comparison"
# "transformer_variants_comparison" 
# "best_models_final_comparison"
# "ablation_study_results"

# Dedicated final results path - MODIFY THIS TO YOUR PREFERRED PATH
final_results_path = '/mydata/deepcloud/yves/final_results'
plot_output_dir = join(final_results_path, SUBFOLDER)

# Optional: create the directory if it doesn't exist
os.makedirs(plot_output_dir, exist_ok=True)

# Define models to evaluate
models = [
    # those were the old models with distinct performance k=4 and 10^-6 clamp, old trainng setup
    #{'name': 'GNN-3D-64-L2-100-new', 'path': '/mydata/deepcloud/yves/results-temp/gnn3d_id39_tendency_64_l2_100_new/test'},
    #{'name': 'GNN-1D-64-L2-100', 'path': '/mydata/deepcloud/yves/results-temp/gnn_64_l2_tendency_1d_triangle_fully_connected/test'},
        
        
    #{'name': 'GNN-3D-64-L2-100-NoFully', 'path': '/mydata/deepcloud/yves/results-temp/gnn3d_id39_tendency_64_l2_100_nofully/test'},
    #{'name': 'GNN-3D-64-L2-100-Indep', 'path': '/mydata/deepcloud/yves/results-temp/gnn3d_id39_tendency_64_l2_indep_100/test'},
    
    # now want to campare old vs new for 1d with clamp 10^-20 and k=4, to see if clamp is the problem, can check training setup by comparing k=4 with 10^-20 for old vs new 1d
    
    
    # Here without dropouts and k1 standardization and clamp 10^-20, 3d and 1 d look alsmot the same
    {'name': 'GNN-1D-64-L2-100', 'path': '/mydata/deepcloud/yves/results-new/gnn_1d_tendency_64_l2_100_k1/test'},
    {'name': 'GNN-3D-64-L2-100', 'path': '/mydata/deepcloud/yves/results-new/gnn_3d_tendency_64_l2_100_k1/test'},
       
    # Here with correct variance values during rescaling clamp 10^-20 and k=4? had a dropout issue 0.3, dropout might have make the diference diminishing
    #{'name': 'GNN-1D-64-L2-100', 'path': '/mydata/deepcloud/yves/results-new/gnn_1d_tendency_64_l2_100/test'},
    #{'name': 'GNN-3D-64-L2-100', 'path': '/mydata/deepcloud/yves/results-new/gnn_3d_tendency_64_l2_100/test'},
    
    # Here with 4sigma std
    # {'name': 'GNN-3D-4-sigma', 'path': '/mydata/deepcloud/yves/results-new/gnn_3d_tendency_64_l2_100_4sig/test'},
        
   # gt used k=4 and clamp 10^-20?    
   #{'name': 'GT-3D-enha-64-L4-MR4-100', 'path': '/mydata/deepcloud/yves/results-new/gt_enhanced_64_l4_drop03_triangle39_k2/test'},
   #{'name': 'GT-3D-simp-64-L4-MR2-100', 'path': '/mydata/deepcloud/yves/results-new/gt_simplified_64_l4_drop03_triangle39_k1/test'},
   #{'name': 'GT-2D-genc-1024-L2-MR2-100', 'path': '/mydata/deepcloud/yves/results-new/gt_gencast_1024_l2_triangle39_k2_new/test'},  
    
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

    # Handle different model architectures
    # GNN-3D has shape [samples, spatial_points, height, features]
    # GNN-1D has shape [samples, height, features]
    if y_true.ndim == 4:  # GNN-3D case
        print(f'Detected 4D tensor (GNN-3D), averaging over spatial dimension')
        # Average over both sample and spatial dimensions
        if y_true.shape[0] > 1:
            y_true_h_tmp = torch.mean(y_true, dim=(0, 1))  # Average over samples and spatial points
            y_pred_h_tmp = torch.mean(y_pred, dim=(0, 1))
        else:
            y_true_h_tmp = torch.mean(y_true.squeeze(0), dim=0)  # Remove sample dim, average over spatial
            y_pred_h_tmp = torch.mean(y_pred.squeeze(0), dim=0)
    elif y_true.ndim == 3:  # GNN-1D case
        print(f'Detected 3D tensor (GNN-1D), processing normally')
        if y_true.shape[0] > 1:
            y_true_h_tmp = torch.mean(y_true, dim=0)  # Average over samples
            y_pred_h_tmp = torch.mean(y_pred, dim=0)
        else:
            y_true_h_tmp = y_true.squeeze(0)  # Remove sample dimension
            y_pred_h_tmp = y_pred.squeeze(0)
    else:
        raise ValueError(f"Unexpected tensor dimensions: {y_true.ndim}")

    print(f'Final shape for plotting: {y_true_h_tmp.shape}')

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
    "Temp. Tendency": "K s⁻¹", 
    "Temp. Tend. (Dyn.)": "K s⁻¹",
    "U-Wind Tendency": "m s⁻²",
    "V-Wind Tendency": "m s⁻²",
    "Humidity Tend.": "kg m⁻³ s⁻¹",
    "Cloud Water Tend.": "kg m⁻³ s⁻¹",
    "Cloud Ice Tend.": "kg m⁻³ s⁻¹"
}

# Optional: Full names for reference
target_units_full = {
    "Sum of Temperature Tendency": "K s⁻¹", 
    "Dynamical Temperature Tendency": "K s⁻¹",
    "Sum of Zonal Wind Tendency": "m s⁻²",
    "Sum of Meridional Wind Tendency": "m s⁻²",
    "Convective Tend. Absolute Humidity": "kg m⁻³ s⁻¹",
    "Convective Tend. Cloud Water Mass Density": "kg m⁻³ s⁻¹",
    "Convective Tend. Cloud Ice Mass Density": "kg m⁻³ s⁻¹"
}

# Height level to kilometer mapping (approximate values for 70 levels)
# You may need to adjust these values based on your specific model's vertical grid
height_km = {
    0: 65,
    10: 39,
    20: 25,
    30: 15,
    40: 8,
    50: 4,
    60: 1,
    70: 0,
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
height_range = np.arange(y_true_h.shape[0])

# Create figure with 2-row layout for publication
fig = plt.figure(figsize=(8, 7.5))  # Match exact dimensions from MAE script

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
    Plots true vs predicted lines for each model on the same subplot with publication styling.
    """
    ax = fig.add_subplot(*subplot_pos)
    
    # Define colors for consistent plotting
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink']
    
    # Plot TRUE (black solid)
    ax.plot(
        true_data[:, channel_idx],
        height_vals,
        label='True' if channel_idx == 0 else None,
        color='black',
        linewidth=1.5,  # Thinner lines
        linestyle='-',  # Solid line for true values
    )

    # Define line styles for different models
    line_styles = ['-', '--', '-.', ':', '--', ':']  # Different styles for each model

    # Plot PREDICTIONS for each model
    for i, (pred_data, model_name) in enumerate(zip(pred_data_list, models_names)):
        ax.plot(
            pred_data[:, channel_idx],
            height_vals,
            label=f'Pred • {model_name}' if channel_idx == 0 else None,
            color=colors[i % len(colors)],
            linestyle=line_styles[i % len(line_styles)],  # Cycle through line styles
            linewidth=1.5,  # Thinner lines
        )

    # Lighter grid
    ax.grid(True, alpha=0.5, linewidth=0.5)  # Reduced alpha and linewidth for lighter grid
    ax.invert_yaxis()
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    # Only show y-axis ticks on leftmost plots (indices 0 and 4)
    if channel_idx not in [0, 4]:
        ax.set_yticklabels([])  # Remove y-axis tick labels
        ax.tick_params(axis='y', which='both', length=0)  # Remove tick marks
    else:
        # Set custom y-axis labels with height level and km for leftmost plots
        yticks = np.arange(0, 71, 10)  # 0, 10, 20, ..., 70
        ax.set_yticks(yticks)
        
        # Create labels with format "level (km)"
        yticklabels = []
        for tick in yticks:
            if tick in height_km:
                yticklabels.append(f'{tick} ({height_km[tick]:.0f} km)')
            else:
                # Interpolate if exact value not in mapping
                yticklabels.append(f'{tick}')
        
        ax.set_yticklabels(yticklabels, fontsize=8)

    if title:
        ax.set_title(title, fontsize=10, pad=2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    
    # Setup scientific notation with proper formatting
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.xaxis.set_major_formatter(formatter)
    
    # Adjust the offset text position
    ax.xaxis.get_offset_text().set_fontsize(7)  # Smaller font for offset
    ax.xaxis.labelpad = 10  # Extra padding for x-axis label
    
    # Set appropriate number of ticks for narrow plots
    ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(8))
    
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

# Create one subplot for each of the 7 target channels in a 2-row layout
ax_list = []

for i, (label, units) in enumerate(target_units.items()):
    # First 4 plots in row 1, last 3 plots in row 2
    if i < 4:
        subplot_idx = (2, 4, i + 1)  # Row 1: 4 plots
    else:
        subplot_idx = (2, 4, i + 1)  # Row 2: continues numbering

    ax = add_subplot_true_pred(
        fig=fig,
        height_vals=height_range,
        true_data=y_true_h,
        pred_data_list=y_pred_hs,
        subplot_pos=subplot_idx,
        models_names=models_name,
        channel_idx=i,
        title=label,
        ylabel='Height index' if i in [0, 4] else None,  # Show ylabel on first plot of each row
        xlabel=f'[{units}]'  # Match MAE script format
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
           bbox_to_anchor=(0.5, -0.03), frameon=False, fontsize=9)

# Tight layout with extra space at bottom for legend
fig.tight_layout(rect=[0, 0.02, 1, 0.98], pad=0.1, h_pad=1, w_pad=0.5)

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

output_path = join(plot_output_dir, f"{PLOT_NAME}.png")
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"Saved MAE visualization to {output_path}")

plt.show()


