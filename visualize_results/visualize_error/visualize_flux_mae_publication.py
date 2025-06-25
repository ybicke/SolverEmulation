import torch
import pickle
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
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
    'grid.alpha': 0.6,         # Lighter grid
    'grid.linewidth': 0.5,     # Thinner grid lines
    'axes.grid': True,         # Enable grid by default
    'axes.axisbelow': True,    # Grid behind plot elements
})

# Configuration
save_to_test_path = False

# === PLOT CONFIGURATION ===
# Manually specify plot name and subfolder
PLOT_NAME = "GNN_vs_ViT"  # Change this for each plot scenario
SUBFOLDER = "MAE_Flux"  # Options: "flux_MAE", "true_vs_predicted", or any custom folder name

# Examples of plot names for different scenarios:
# "AFNO_vs_BiLSTM_flux_comparison"
# "best_flux_models_final"
# "ablation_study_flux_results"

final_results_path = '/mydata/deepcloud/yves/results_final'
plot_output_dir = join(final_results_path, SUBFOLDER)
os.makedirs(plot_output_dir, exist_ok=True)



models = [
    {'name': 'GNN-128-l4', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results/gnn_128_l4/test'},
    {'name': 'ViT-128-l4', 'path': '/mydata/deepcloud/yves/results_git/vit_column_128_l4_h6_concat/test'},
    
    # Add more models as needed
]

y_mae_hs = []
h_mae_hs = []

for model in models:
    test_path = model['path']
    print(f'\nProcessing model: {model["name"]}')

    with open(join(test_path, 'y_true.pickle'), 'rb') as handle:
        y_true = pickle.load(handle)
    with open(join(test_path, 'y_pred.pickle'), 'rb') as handle:
        y_pred = pickle.load(handle)
    with open(join(test_path, 'h_true.pickle'), 'rb') as handle:
        h_true = pickle.load(handle)
    with open(join(test_path, 'h_pred.pickle'), 'rb') as handle:
        h_pred = pickle.load(handle)

    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred)
    if isinstance(h_true, np.ndarray):
        h_true = torch.tensor(h_true)
    if isinstance(h_pred, np.ndarray):
        h_pred = torch.tensor(h_pred)
    
    # Print shapes of loaded data
    print(f'  y_true shape: {y_true.shape}')
    print(f'  y_pred shape: {y_pred.shape}')
    print(f'  h_true shape: {h_true.shape}')
    print(f'  h_pred shape: {h_pred.shape}')
    
    print('  calculating errors...')
    # Handle different data formats
    dim = [0, 1] if len(y_true.shape) == 4 else 0
    y_mae_h = torch.mean(torch.abs(y_true - y_pred), dim=dim)
    h_mae_h = torch.mean(torch.abs(h_true - h_pred), dim=dim)
    
    # Print shape of MAE data
    print(f'  y_mae_h shape: {y_mae_h.shape}')
    print(f'  h_mae_h shape: {h_mae_h.shape}')
    
    # Flip to match height convention (surface at bottom) - same as original visualize.py
    y_mae_hs.append(torch.flip(y_mae_h, [0]))
    h_mae_hs.append(torch.flip(h_mae_h, [0]))

# Short descriptive titles for publication - following same pattern as tendency script
flux_target_units = {
    "LW Upward": "W m⁻²",
    "LW Downward": "W m⁻²",
    "SW Upward": "W m⁻²", 
    "SW Downward": "W m⁻²",
    "LW Heating": "K day⁻¹",
    "SW Heating": "K day⁻¹"
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
                    channel_idx, data_type='y', title=None, ylabel=None, xlabel=None):
    """
    Plots MAE lines for each model on the same subplot with publication styling.
    """
    ax = fig.add_subplot(*subplot_pos)
    
    # Define colors for consistent plotting
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink']
    
    # Define line styles for different models
    line_styles = ['--', '-', '-.', '--', '-.', '--']  # Different styles for each model
    
    # Plot MAE for each model
    for i, (mae_data, model_name) in enumerate(zip(mae_data_list, models_names)):
        # Adjust height vals based on data type (heating has 70 levels, flux has 71)
        if data_type == 'h':
            plot_height = height_vals[:-1]  # Remove last level for heating data (70 levels)
        else:
            plot_height = height_vals  # Full range for flux data (71 levels)
            
        ax.plot(
            mae_data[:, channel_idx],
            plot_height,
            label=model_name if channel_idx == 0 else None,
            color=colors[i % len(colors)],
            linestyle=line_styles[i % len(line_styles)],  # Cycle through line styles
            linewidth=1.5,  # Thinner lines
        )

    # Lighter grid
    ax.grid(True, alpha=0.5, linewidth=0.5)  # Reduced alpha and linewidth for lighter grid
    ax.invert_yaxis()
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    # Only show y-axis ticks on leftmost plots (indices 0 and 3 for 2x3 layout)
    if channel_idx not in [0, 3]:
        ax.set_yticklabels([])  # Remove y-axis tick labels
        ax.tick_params(axis='y', which='both', length=0)  # Remove tick marks
    else:
        # Set custom y-axis labels with height level and km for leftmost plots
        if data_type == 'h':
            yticks = np.arange(0, 70, 10)  # 0, 10, 20, ..., 60 for heating rates
        else:
            yticks = np.arange(0, 71, 10)  # 0, 10, 20, ..., 70 for fluxes
        
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
height_range = np.arange(71)  # 0 to 70 for flux levels (71 total)

# Create figure with 2-row layout for publication
fig_mae = plt.figure(figsize=(10, 7.5))
ax_list_mae = []

# Create one subplot for each of the 6 flux channels in a 2-row layout
# Following the pattern from original visualize.py plotting order
flux_channels = [
    (1, 'y', "LW Downward"),   # y[:, 1] - position (2,3,1)
    (0, 'y', "LW Upward"),     # y[:, 0] - position (2,3,2)  
    (3, 'y', "SW Downward"),   # y[:, 3] - position (2,3,3)
    (2, 'y', "SW Upward"),     # y[:, 2] - position (2,3,4)
    (0, 'h', "LW Heating"),    # h[:, 0] - position (2,3,5)
    (1, 'h', "SW Heating"),    # h[:, 1] - position (2,3,6)
]

for i, (channel_idx, data_type, label) in enumerate(flux_channels):
    # 2 rows, 3 columns
    subplot_idx = (2, 3, i + 1)
    
    # Select appropriate data list
    data_list = y_mae_hs if data_type == 'y' else h_mae_hs

    ax = add_subplot_mae(
        fig=fig_mae,
        height_vals=height_range,
        mae_data_list=data_list,
        subplot_pos=subplot_idx,
        models_names=models_name,
        channel_idx=channel_idx,
        data_type=data_type,
        title=label,
        ylabel='Height index' if i in [0, 3] else None,
        xlabel=f'MAE [{flux_target_units[label]}]'
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
               bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=9)

# Tight layout with extra space at bottom for legend
fig_mae.tight_layout(rect=[0, 0.05, 1, 0.98], pad=0.1, h_pad=1.5, w_pad=0.8)

# Save the figure
# Final output: /mydata/deepcloud/yves/final_results/{SUBFOLDER}/{PLOT_NAME}.png
# To change: modify PLOT_NAME and SUBFOLDER at the top of the script
output_path = join(plot_output_dir, f"{PLOT_NAME}.png")
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"Saved flux MAE visualization to {output_path}")

plt.show() 