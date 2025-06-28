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
PLOT_NAME = "flux1d_linear_log"  # Change this for each plot scenario
SUBFOLDER = "MAE_Flux"  # Options: "flux_MAE", "true_vs_predicted", or any custom folder name

# Examples of plot names for different scenarios:
# "AFNO_vs_BiLSTM_flux_comparison"
# "best_flux_models_final"
# "ablation_study_flux_results"

final_results_path = '/mydata/deepcloud/yves/results_final'
plot_output_dir = join(final_results_path, SUBFOLDER)
os.makedirs(plot_output_dir, exist_ok=True)

models = [
    # Fluxes 1D models
    {'name': 'GNN-128-l4', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results/gnn_128_l4/test'},
    {'name': 'ViT-128-l4', 'path': '/mydata/deepcloud/yves/results_git/vit_column_128_l4_h6_concat/test'},
    {'name': 'AFNO-128-l4', 'path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_clean/test'},
    {'name': 'BiLSTM-medium', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results/rnn_medium/test'},

    # Fluxes 1D models hrlu
    # {'name': 'AFNO-128-hrlu', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results/afno_1d_128_hrlu/test'},
    #{'name': 'GNN-128-hrlu', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results/gnn_1d_128_hrlu_0005/test'},
    # {'name': 'BiLSTM-32-128-hrlu', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results/rnn_32_128_hrlu_0005/test'},
    #{'name': 'ViT-128-hrlu', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results/vit_128_hrlu_0005_new/test'},
]

y_mae_hs, h_mae_hs = [], []

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
    
    print(f'  y_true shape: {y_true.shape}')
    print(f'  y_pred shape: {y_pred.shape}')
    print(f'  h_true shape: {h_true.shape}')
    print(f'  h_pred shape: {h_pred.shape}')

    print('  calculating errors...')
    # Handle different data formats
    dim = [0, 1] if len(y_true.shape) == 4 else 0
    y_mae_h = torch.mean(torch.abs(y_true - y_pred), dim=dim)
    h_mae_h = torch.mean(torch.abs(h_true - h_pred), dim=dim)

    # Flip to match height convention (surface at bottom) - same as original visualize.py
    y_mae_hs.append(torch.flip(y_mae_h, [0]))
    h_mae_hs.append(torch.flip(h_mae_h, [0]))

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

def add_subplot_flux(fig, x, ys, subplot_pos, models_name, xlabel=None, ylabel=None, 
                     title=None, is_heating=False, is_flux=False):
    """
    Publication-ready subplot function based on original visualize.py but with better styling.
    """
    ax = fig.add_subplot(*subplot_pos)
    
    # Define colors and line styles for consistent plotting
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink']
    line_styles = ['--', '-.', '-', '-.', '-', '--']  # Different styles for each model
    
    # Plot each model
    for i, (y, model_name) in enumerate(zip(ys, models_name)):
        ax.plot(
            y, 
            x, 
            label=model_name,
            color=colors[i % len(colors)],
            linestyle=line_styles[i % len(line_styles)],
            linewidth=1.5
        )
    
    # Basic styling (grid handled below for log scale)
    ax.invert_yaxis()
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    # Y-axis labels - only show on leftmost plots (positions 1 and 4 in 2x3 grid)
    subplot_idx = subplot_pos[2]  # Position in grid (1-6)
    if subplot_idx not in [1, 4]:
        ax.set_yticklabels([])  # Remove y-axis tick labels
        ax.tick_params(axis='y', which='both', length=0)  # Remove tick marks
    else:
        # Set custom y-axis labels with height level and km for leftmost plots
        if is_heating:
            yticks = np.arange(0, 69, 10)  # 0, 10, 20, ..., 60 for heating rates (69 levels)
        else:
            yticks = np.arange(0, 71, 10)  # 0, 10, 20, ..., 70 for fluxes
        
        ax.set_yticks(yticks)
        
        # Create labels with format "level (km)"
        yticklabels = []
        for tick in yticks:
            if tick in height_km:
                yticklabels.append(f'{tick} ({height_km[tick]:.0f} km)')
            else:
                yticklabels.append(f'{tick}')
        
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
    
    # Use different scaling based on data type
    if is_flux:
        # Linear scale for flux data (more interpretable MAE values)
        ax.set_xscale('linear')
        ax.set_xlim(-0.5, 7)  # Flux data: -0.2 to 8 W m⁻² 
        
        # Set up linear ticks for fluxes
        ax.xaxis.set_major_locator(ticker.MaxNLocator(6))  # Maximum 6 major ticks
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())  # Auto minor ticks
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        
        # Enable grid for both major and minor ticks
        ax.grid(True, which='major', alpha=0.6, linewidth=0.5, color='gray')
        #ax.grid(True, which='minor', alpha=0.3, linewidth=0.3, color='gray')
    else:
        # Log scale for heating rates (handle large dynamic range)
        ax.set_xscale('log')
        ax.set_xlim(1e-2, 1e3)  # Heating rates: 10^-2 to 10^2 K day⁻¹
        ax.set_xlim(-0.5, 100)  # Flux data: -0.2 to 8 W m⁻² 

        
        # Set up log ticks for heating rates
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=6))
        ax.xaxis.set_minor_locator(ticker.NullLocator())  # No minor ticks for log scale
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'10$^{{{int(np.log10(x))}}}$'))
        
        #ax.xaxis.set_major_locator(ticker.MaxNLocator(6))  # Maximum 6 major ticks
        #ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())  # Auto minor ticks
        #ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        
        # Enable grid only for major ticks (powers of 10)
        ax.grid(True, which='major', alpha=0.6, linewidth=0.5, color='gray')
        ax.grid(False, which='minor')  # No minor grid lines for log scale
    
    # Adjust positioning
    ax.xaxis.labelpad = 10
    
    # Set y-axis ticks
    ax.yaxis.set_major_locator(ticker.MaxNLocator(8))
    
    return ax

# Prepare data
models_name = [model['name'] for model in models]
mask = [True] * len(models_name)

# Create figure with proper aspect ratio for 2x3 layout
fig = plt.figure(figsize=(6.5, 8))  # Much narrower figure for 3 columns
ax_list = []

# Create subplots following original visualize.py order
# Row 1: Shortwave - Downward, Upward, Heating
ax1 = add_subplot_flux(fig, x=range(71), ys=[y[:, 3] for y in y_mae_hs], 
                       subplot_pos=(2, 3, 1), models_name=models_name, 
                       title='SW Downward flux', xlabel='MAE [W m⁻²]', is_flux=True)
ax_list.append(ax1)

ax2 = add_subplot_flux(fig, x=range(71), ys=[y[:, 2] for y in y_mae_hs], 
                       subplot_pos=(2, 3, 2), models_name=models_name, 
                       title='SW Upward flux', xlabel='MAE [W m⁻²]', is_flux=True)
ax_list.append(ax2)

ax3 = add_subplot_flux(fig, x=range(69), ys=[h[:69, 1] for h in h_mae_hs], 
                       subplot_pos=(2, 3, 3), models_name=models_name, 
                       title='SW Heating rates', xlabel='MAE [K day⁻¹]', is_heating=True)
ax_list.append(ax3)

# Row 2: Longwave - Downward, Upward, Heating
ax4 = add_subplot_flux(fig, x=range(71), ys=[y[:, 1] for y in y_mae_hs], 
                       subplot_pos=(2, 3, 4), models_name=models_name, 
                       xlabel='MAE [W m⁻²]', title='LW Downward flux', is_flux=True)
ax_list.append(ax4)

ax5 = add_subplot_flux(fig, x=range(71), ys=[y[:, 0] for y in y_mae_hs], 
                       subplot_pos=(2, 3, 5), models_name=models_name, 
                       xlabel='MAE [W m⁻²]', title='LW Upward flux', is_flux=True)
ax_list.append(ax5)

ax6 = add_subplot_flux(fig, x=range(69), ys=[h[:69, 0] for h in h_mae_hs], 
                       subplot_pos=(2, 3, 6), models_name=models_name, 
                       xlabel='MAE [K day⁻¹]', title='LW Heating rates', is_heating=True)
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
           bbox_to_anchor=(0.5, -0.03), frameon=False, fontsize=9)

# Tight layout with extra space at top and bottom for titles and legend
fig.tight_layout(rect=[0, 0.05, 1, 0.92], pad=0.3, h_pad=2.0, w_pad=1.0)

# Save the figure
output_path = join(plot_output_dir, f"{PLOT_NAME}.png")
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"Saved flux MAE visualization to {output_path}")

plt.show() 