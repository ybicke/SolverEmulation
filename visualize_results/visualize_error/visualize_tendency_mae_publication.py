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

# Configuration
save_to_test_path = False

# === PLOT CONFIGURATION ===
# Manually specify plot name and subfolder
PLOT_NAME = "GNN_1D_vs_GNN_3D_no_dropout"  # Change this for each plot scenario
SUBFOLDER = "MAE"  # Options: "MAE", "true_vs_predicted", or any custom folder name

# Examples of plot names for different scenarios:
# "3D_vs_1D_GNN_comparison"
# "transformer_variants_comparison" 
# "best_models_final_comparison"
# "ablation_study_results"

final_results_path = '/mydata/deepcloud/yves/final_results'
plot_output_dir = join(final_results_path, SUBFOLDER)
os.makedirs(plot_output_dir, exist_ok=True)

models = [
    #{'name': 'GNN-1D-64-L2-100', 'path': '/mydata/deepcloud/yves/results-temp/gnn_64_l2_tendency_1d_triangle_fully_connected/test'},
    #{'name': 'ViT-1D-64-L2-100', 'path': '/mydata/deepcloud/yves/results-temp/vit_128_l4_tendency_1d_triangle/test'},
    
    # Here with correct variance values during rescaling
    #{'name': 'GNN-1D-64-L2-100', 'path': '/mydata/deepcloud/yves/results-new/gnn_1d_tendency_64_l2_100/test'},
    #{'name': 'GNN-3D-64-L2-100', 'path': '/mydata/deepcloud/yves/results-new/gnn_3d_tendency_64_l2_100/test'},
    
    #Here without dropouts and k1 standardization
    {'name': 'GNN-1D-64-L2-100', 'path': '/mydata/deepcloud/yves/results-new/gnn_1d_tendency_64_l2_100_k1/test'},
    {'name': 'GNN-3D-64-L2-100', 'path': '/mydata/deepcloud/yves/results-new/gnn_3d_tendency_64_l2_100_k1/test'},
    
    # Here with 4sigma std
    # {'name': 'GNN-3D-4-sigma', 'path': '/mydata/deepcloud/yves/results-new/gnn_3d_tendency_64_l2_100_4sig/test'},



    #{'name': 'GNN-3D-64-L2-100-old', 'path': '/mydata/deepcloud/yves/results-temp/gnn3d_id39_tendency_64_l2_100_new/test'},
    #{'name': 'GNN-1D-64-L2-100-old', 'path': '/mydata/deepcloud/yves/results-temp/gnn_64_l2_tendency_1d_triangle_fully_connected/test'},
    
    #{'name': 'GT-3D-enha-64-L4-MR4-100', 'path': '/mydata/deepcloud/yves/results-new/gt_enhanced_64_l4_drop03_triangle39_k2/test'},
    #{'name': 'GT-3D-simp-64-L4-MR2-100', 'path': '/mydata/deepcloud/yves/results-new/gt_simplified_64_l4_drop03_triangle39_k1/test'},
    #{'name': 'GT-2D-genc-1024-L2-MR2-100', 'path': '/mydata/deepcloud/yves/results-new/gt_gencast_1024_l2_triangle39_k2_new/test'},

    # Add more models as needed
]

y_mae_hs = []

for model in models:
    test_path = model['path']
    print(f'\nProcessing model: {model["name"]}')

    with open(join(test_path, 'y_true.pickle'), 'rb') as handle:
        y_true = pickle.load(handle)
    with open(join(test_path, 'y_pred.pickle'), 'rb') as handle:
        y_pred = pickle.load(handle)

    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred)
    
    # Handle different data formats
    if len(y_true.shape) == 4:  # Shape: [samples, areas, height, features]
        y_mae_h = torch.mean(torch.abs(y_true - y_pred), dim=(0, 1))
    else:  # Shape: [batch, height, features]
        y_mae_h = torch.mean(torch.abs(y_true - y_pred), dim=0)
    
    y_mae_hs.append(y_mae_h)

# Short descriptive titles for publication
target_units = {
    "Temp. Tendency": "K s⁻¹", 
    "Temp. Tend. (Dyn.)": "K s⁻¹",
    "U-Wind Tendency": "m s⁻²",
    "V-Wind Tendency": "m s⁻²",
    "Humidity Tend.": "kg m⁻³ s⁻¹",
    "Cloud Water Tend.": "kg m⁻³ s⁻¹",
    "Cloud Ice Tend.": "kg m⁻³ s⁻¹"
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

def add_subplot_mae(fig, height_vals, mae_data_list, subplot_pos, models_names, 
                    channel_idx, title=None, ylabel=None, xlabel=None):
    """
    Plots MAE lines for each model on the same subplot with publication styling.
    """
    ax = fig.add_subplot(*subplot_pos)
    
    # Define colors for consistent plotting
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink']
    
    # Define line styles for different models
    line_styles = ['-', '--', '-.', ':', '--', ':']  # Different styles for each model
    
    # Plot MAE for each model
    for i, (mae_data, model_name) in enumerate(zip(mae_data_list, models_names)):
        ax.plot(
            mae_data[:, channel_idx],
            height_vals,
            label=model_name if channel_idx == 0 else None,
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
    data_values = []
    for mae in mae_data_list:
        data_values.append(mae[:, channel_idx])
    
    min_val = min(d.min().item() for d in data_values)
    max_val = max(d.max().item() for d in data_values)
    
    # Add 5% padding to the range
    range_val = max_val - min_val
    min_val = min_val - 0.05 * range_val
    max_val = max_val + 0.05 * range_val
    
    # Set the limits
    ax.set_xlim(min_val, max_val)
    
    return ax

# Prepare data
models_name = [model['name'] for model in models]
height_range = np.arange(y_mae_hs[0].shape[0])

# Create figure with 2-row layout for publication
fig_mae = plt.figure(figsize=(8, 7.5))
ax_list_mae = []

# Create one subplot for each of the 7 target channels in a 2-row layout
for i, (label, units) in enumerate(target_units.items()):
    # First 4 plots in row 1, last 3 plots in row 2
    if i < 4:
        subplot_idx = (2, 4, i + 1)
    else:
        subplot_idx = (2, 4, i + 1)

    ax = add_subplot_mae(
        fig=fig_mae,
        height_vals=height_range,
        mae_data_list=y_mae_hs,
        subplot_pos=subplot_idx,
        models_names=models_name,
        channel_idx=i,
        title=label,
        ylabel='Height index' if i in [0, 4] else None,
        xlabel=f'MAE [{units}]'
    )
    
    ax_list_mae.append(ax)

# Create a single legend at the bottom
handles, labels = [], []
for ax in ax_list_mae:
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in labels:
            handles.append(h)
            labels.append(l)

# Place legend below all subplots
fig_mae.legend(handles, labels, loc='lower center', ncol=len(labels),
               bbox_to_anchor=(0.5, -0.03), frameon=False, fontsize=9)

# Tight layout with extra space at bottom for legend
fig_mae.tight_layout(rect=[0, 0.02, 1, 0.98], pad=0.1, h_pad=1, w_pad=0.5)

# Save the figure
# Final output: /mydata/deepcloud/yves/final_results/{SUBFOLDER}/{PLOT_NAME}.png
# To change: modify PLOT_NAME and SUBFOLDER at the top of the script
output_path = join(plot_output_dir, f"{PLOT_NAME}.png")
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"Saved MAE visualization to {output_path}")

plt.show()