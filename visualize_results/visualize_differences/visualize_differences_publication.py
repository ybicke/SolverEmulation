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

# === PLOT CONFIGURATION ===
# Manually specify plot name and subfolder
PLOT_NAME = "AFNO"  # Change this for each plot scenario
SUBFOLDER = "Differences_Flux"  # Output subfolder

final_results_path = '/mydata/deepcloud/yves/results_final'
plot_output_dir = join(final_results_path, SUBFOLDER)
os.makedirs(plot_output_dir, exist_ok=True)

# Define the models to compare
models = [
    #{'name': 'GNN-128-l4', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results/gnn_128_l4/test'},
    #{'name': 'ViT-128-l4', 'path': '/mydata/deepcloud/yves/results_git/vit_column_128_l4_h6_concat/test'},
    {'name': 'AFNO-128-l4', 'path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_clean/test'},
    #{'name': 'BiLSTM-medium', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results/rnn_medium/test'},
]

# Sample selection
sample_index = 91  # Change this to select a different sample

# Vertical level range - upper 30 levels (0 = uppermost, 70 = surface)
upper_level = 0  # Uppermost level
lower_level = 70  # Upper 30 levels (0-29)
print(f"Visualizing vertical differences for sample {sample_index}, upper {lower_level-upper_level} levels ({upper_level} to {lower_level-1})")

# Height level to kilometer mapping (approximate values for 70 levels)
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

# ====================================================================
# DATA LOADING AND PROCESSING
# ====================================================================

# Load data for all models
y_true_list, y_pred_list, h_true_list, h_pred_list = [], [], [], []
y_diff_true = None  # Store true differences (same for all models)
y_diff_models, h_diff_models = [], []

for i, model in enumerate(models):
    test_path = model['path']
    print(f'\nProcessing model: {model["name"]}')
    
    # Load flux data
    with open(join(test_path, 'y_true.pickle'), 'rb') as handle:
        y_true = pickle.load(handle)
    with open(join(test_path, 'y_pred.pickle'), 'rb') as handle:
        y_pred = pickle.load(handle)
    
    # Load heating rate data
    with open(join(test_path, 'h_true.pickle'), 'rb') as handle:
        h_true = pickle.load(handle)
    with open(join(test_path, 'h_pred.pickle'), 'rb') as handle:
        h_pred = pickle.load(handle)
    
    # Convert to tensors
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true, dtype=torch.float32)
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
    if isinstance(h_true, np.ndarray):
        h_true = torch.tensor(h_true, dtype=torch.float32)
        h_pred = torch.tensor(h_pred, dtype=torch.float32)
    
    print(f'  Flux shapes: y_true={y_true.shape}, y_pred={y_pred.shape}')
    print(f'  Heating shapes: h_true={h_true.shape}, h_pred={h_pred.shape}')
    
    # Calculate vertical differences for the selected sample
    # To get levels 0-29 (upper 30 levels), slice from upper_level to lower_level
    y_true_sample = y_true[sample_index, upper_level:lower_level, :]
    y_pred_sample = y_pred[sample_index, upper_level:lower_level, :]
    h_true_sample = h_true[sample_index, upper_level:lower_level-1, :] 
    h_pred_sample = h_pred[sample_index, upper_level:lower_level-1, :]
    
    # Calculate differences (gradients) between adjacent levels
    y_diff_pred = torch.diff(y_pred_sample, dim=0)  # Shape: [levels-1, variables]
    h_diff_pred = torch.diff(h_pred_sample, dim=0)  # Shape: [levels-2, variables] 
    
    # Store true differences only once (same for all models)
    if y_diff_true is None:
        y_diff_true = torch.diff(y_true_sample, dim=0)
        h_diff_true = torch.diff(h_true_sample, dim=0)
    
    # Store differences - flip to match the axis labels (level 0 at top)
    y_diff_models.append(torch.flip(y_diff_pred, [0]))
    h_diff_models.append(torch.flip(h_diff_pred, [0]))

# Flip true differences as well to match the axis labels
y_diff_true = torch.flip(y_diff_true, [0])
h_diff_true = torch.flip(h_diff_true, [0])

print(f"Calculated differences - y_diff shape: {y_diff_true.shape}, h_diff shape: {h_diff_true.shape}")

# ====================================================================
# PLOTTING FUNCTIONS
# ====================================================================

def add_subplot_differences(fig, x, y_true_diff, y_pred_diffs, subplot_pos, models_name, 
                           xlabel=None, ylabel=None, title=None, is_heating=False):
    """
    Publication-ready subplot function for vertical differences.
    """
    ax = fig.add_subplot(*subplot_pos)
    
    # Define colors and line styles for consistent plotting
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink']
    line_styles = ['-', '-', '-.', '--', '-.', '--']  # Different styles for each model
    
    # Plot predicted differences for each model
    for i, (y_diff, model_name) in enumerate(zip(y_pred_diffs, models_name)):
        ax.plot(
            y_diff, 
            x, 
            label=model_name,
            color=colors[i % len(colors)],
            linestyle=line_styles[i % len(line_styles)],
            linewidth=1.5,
            alpha=0.8
        )
    
    # Plot true differences
    ax.plot(
        y_true_diff, 
        x, 
        label='True',
        color='black',
        linestyle=':',
        linewidth=1.5,
        alpha=1.0
    )
    
    # Basic styling
    ax.invert_yaxis()
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    # Y-axis labels - only show on leftmost plots (positions 1 and 4 in 2x3 grid)
    subplot_idx = subplot_pos[2]  # Position in grid (1-6)
    if subplot_idx not in [1, 4]:
        ax.set_yticklabels([])  # Remove y-axis tick labels
        ax.tick_params(axis='y', which='both', length=0)  # Remove tick marks
    else:
        # Set custom y-axis labels every 10 levels starting from 0
        max_tick = len(x)
        yticks = np.arange(0, 71, 10)  # Explicitly include all levels up to 70
        
        # Ensure we don't exceed the data range
        yticks = yticks[yticks <= 70]  # Keep all ticks up to 70
        
        print(f"Debug: len(x)={len(x)}, max_tick={max_tick}, yticks={yticks}")
        
        # Force set the ticks (disable automatic tick locator)
        ax.set_yticks(yticks)
        
        # Create labels with format "level (km)"
        yticklabels = []
        for tick in yticks:
            # Direct mapping: tick position corresponds to atmospheric level
            actual_level = tick + upper_level
            if actual_level in height_km:
                yticklabels.append(f'{actual_level} ({height_km[actual_level]:.0f} km)')
            else:
                yticklabels.append(f'{actual_level}')
        
        ax.set_yticklabels(yticklabels, fontsize=8)
        
        # Add "Height level" label only to leftmost plots
        ax.set_ylabel('Height level', fontsize=9)
    
    # Labels and title
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10, pad=2)
    
    # Use linear scale for differences (easier to interpret)
    ax.set_xscale('linear')
    
    # Set up linear ticks
    ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    
    # Enable grid
    ax.grid(True, which='major', alpha=0.6, linewidth=0.5, color='gray')
    #ax.grid(True, which='minor', alpha=0.3, linewidth=0.3, color='gray')
    
    # Adjust positioning
    ax.xaxis.labelpad = 10
    
    # Note: y-axis ticks are set manually in the leftmost plots section above
    
    return ax

# ====================================================================
# CREATE PLOTS
# ====================================================================

# Prepare data
models_name = [model['name'] for model in models]

# Create figure with proper aspect ratio for 2x3 layout
fig = plt.figure(figsize=(6.5, 8))  # Slightly wider for difference values
ax_list = []

# Create height level arrays for plotting (differences have one less level)
flux_levels = range(len(y_diff_true))  # levels-1 for flux differences
heating_levels = range(len(h_diff_true))  # levels-2 for heating differences

# Create subplots following original visualize.py order
# Row 1: Shortwave - Downward, Upward, Heating
ax1 = add_subplot_differences(fig, x=flux_levels, 
                             y_true_diff=y_diff_true[:, 3], 
                             y_pred_diffs=[y[:, 3] for y in y_diff_models], 
                             subplot_pos=(2, 3, 1), models_name=models_name, 
                             title='SW Downward flux', 
                             xlabel='Flux Difference [W m⁻²]')
ax_list.append(ax1)

ax2 = add_subplot_differences(fig, x=flux_levels, 
                             y_true_diff=y_diff_true[:, 2], 
                             y_pred_diffs=[y[:, 2] for y in y_diff_models], 
                             subplot_pos=(2, 3, 2), models_name=models_name, 
                             title='SW Upward flux', 
                             xlabel='Flux Difference [W m⁻²]')
ax_list.append(ax2)

ax3 = add_subplot_differences(fig, x=heating_levels, 
                             y_true_diff=h_diff_true[:, 1], 
                             y_pred_diffs=[h[:, 1] for h in h_diff_models], 
                             subplot_pos=(2, 3, 3), models_name=models_name, 
                             title='SW Heating rate', 
                             xlabel='HR Difference [K day⁻¹]', 
                             is_heating=True)
ax_list.append(ax3)

# Row 2: Longwave - Downward, Upward, Heating
ax4 = add_subplot_differences(fig, x=flux_levels, 
                             y_true_diff=y_diff_true[:, 1], 
                             y_pred_diffs=[y[:, 1] for y in y_diff_models], 
                             subplot_pos=(2, 3, 4), models_name=models_name, 
                             xlabel='Flux Difference [W m⁻²]', 
                             title='LW Downward flux')
ax_list.append(ax4)

ax5 = add_subplot_differences(fig, x=flux_levels, 
                             y_true_diff=y_diff_true[:, 0], 
                             y_pred_diffs=[y[:, 0] for y in y_diff_models], 
                             subplot_pos=(2, 3, 5), models_name=models_name, 
                             xlabel='Flux Difference [W m⁻²]', 
                             title='LW Upward flux')
ax_list.append(ax5)

ax6 = add_subplot_differences(fig, x=heating_levels, 
                             y_true_diff=h_diff_true[:, 0], 
                             y_pred_diffs=[h[:, 0] for h in h_diff_models], 
                             subplot_pos=(2, 3, 6), models_name=models_name, 
                             xlabel='HR Difference [K day⁻¹]', 
                             title='LW Heating rate', 
                             is_heating=True)
ax_list.append(ax6)

# Create a single legend at the bottom
handles, labels = [], []
for ax in ax_list:
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in labels:
            handles.append(h)
            labels.append(l)

# Place legend below all subplots
fig.legend(handles, labels, loc='lower center', ncol=len(labels),
           bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=9)

# Tight layout with extra space at top and bottom for titles and legend
fig.tight_layout(rect=[0, 0.05, 1, 0.92], pad=0.3, h_pad=2.0, w_pad=1.0)

# Add overall title
plt.suptitle(f"Sample {sample_index}: Vertical Differences Comparison (Levels {upper_level}-{lower_level-1})", 
             fontsize=12, y=0.95)

# Save the figure
output_path = join(plot_output_dir, f"{PLOT_NAME}_sample_{sample_index}.png")
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"Saved flux differences visualization to {output_path}")

plt.show() 