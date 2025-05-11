import pickle
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from itertools import cycle

# Define the model names and paths
model_names = [
    'gnn_32_l3_optimized',
    'gnn_32_l3_hrlu_005',
    #'gnn_32_l3_hrl_005',
    'gnn_32_l3_hrlu_0005',
]

model_paths = [
    f'/mydata/deepcloud/yves/A_RadiativeFlux/results/{model_names[0]}/test',
    f'/mydata/deepcloud/yves/A_RadiativeFlux/results/{model_names[1]}/test',
    f'/mydata/deepcloud/yves/A_RadiativeFlux/results/{model_names[2]}/test',
]

models = [{'name': name, 'path': path} for name, path in zip(model_names, model_paths)]

# Select a single sample
sample_index = 90  # Change this to select a different sample

# Define vertical level range to visualize
min_level = 0  # Minimum vertical level (inclusive)
max_level = 71  # Maximum vertical level (exclusive)
vertical_levels = range(min_level, max_level)
print(f"Visualizing vertical levels from {min_level} to {max_level-1}")

# Load data
y_true_list, y_pred_list, h_true_list, h_pred_list = [], [], [], []

for model in models:
    test_path = model['path']
    print(f'Loading test files... ({test_path})')
    with open(join(test_path, 'y_true.pickle'), 'rb') as handle:
        y_true = pickle.load(handle)
    with open(join(test_path, 'y_pred.pickle'), 'rb') as handle:
        y_pred = pickle.load(handle)
    with open(join(test_path, 'h_true.pickle'), 'rb') as handle:
        h_true = pickle.load(handle)    
    with open(join(test_path, 'h_pred.pickle'), 'rb') as handle:
        h_pred = pickle.load(handle)
        
    print(f'y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}')
    print(f'h_true shape: {h_true.shape}, h_pred shape: {h_pred.shape}')
    
    y_true_list.append(y_true)
    y_pred_list.append(y_pred)
    h_true_list.append(h_true)
    h_pred_list.append(h_pred)

print(f"Using sample {sample_index} out of {y_true_list[0].shape[0]} total samples")

# Create a figure for vertical differences comparison
fig = plt.figure(figsize=(20, 12))

# Create a helper function to plot differences
def plot_differences(fig, subplot_idx, x_values, y_true, y_preds, models, title, xlabel, ylabel):
    ax = fig.add_subplot(2, 3, subplot_idx)
    color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    
    # Get the specific sample for ground truth
    y_true_sample = y_true[sample_index]
    
    # Calculate the vertical differences for ground truth
    delta_y_true = np.diff(y_true_sample)
    
    # Get x values for differences
    x_diff = np.array(x_values)[:-1]
    
    # Plot differences for each model
    for i, (y_pred, model) in enumerate(zip(y_preds, models)):
        color = next(color_cycle)
        y_pred_sample = y_pred[sample_index]
        delta_y_pred = np.diff(y_pred_sample)
        ax.plot(delta_y_pred, x_diff, linestyle='-', color=color, linewidth=1.5, label=f'{model["name"]}', alpha=0.7)
    
    # Plot ground truth differences
    ax.plot(delta_y_true, x_diff, linestyle=':', color='red', linewidth=2, label='True')
    
    # Set labels and grid
    ax.grid(True, linestyle='-', alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Simple function to invert the y label
    def invert_y_label(y, pos):
        # Convert the position to an integer level
        level = int(round(y))
        # Invert the level number and shift to start at 1 and end at 71
        return str(max_level - 1 - level)
    
    # Apply the formatter to the y-axis
    ax.yaxis.set_major_formatter(FuncFormatter(invert_y_label))
    
    # Create a single legend for the entire figure
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(models) + 1, fontsize="12")
    
    return ax

# Plot vertical differences for both ground truth and model predictions
plot_differences(fig, 1, vertical_levels, 
                y_true_list[0][:, min_level:max_level, 3], 
                [y[:, min_level:max_level, 3] for y in y_pred_list], 
                models, 'Downward Shortwave (Vertical Differences)', 
                'Flux Difference [W/m$^2$]', 'Vertical Level')

plot_differences(fig, 2, vertical_levels, 
                y_true_list[0][:, min_level:max_level, 2], 
                [y[:, min_level:max_level, 2] for y in y_pred_list], 
                models, 'Upward Shortwave (Vertical Differences)', 
                'Flux Difference [W/m$^2$]', 'Vertical Level')

plot_differences(fig, 3, vertical_levels[:-1], 
                h_true_list[0][:, min_level:max_level-1, 1], 
                [h[:, min_level:max_level-1, 1] for h in h_pred_list], 
                models, 'Heating Rates (Shortwave, Vertical Differences)', 
                'Heating Rate Difference [K/day]', 'Vertical Level')

plot_differences(fig, 4, vertical_levels, 
                y_true_list[0][:, min_level:max_level, 1], 
                [y[:, min_level:max_level, 1] for y in y_pred_list], 
                models, 'Downward Longwave (Vertical Differences)', 
                'Flux Difference [W/m$^2$]', 'Vertical Level')

plot_differences(fig, 5, vertical_levels, 
                y_true_list[0][:, min_level:max_level, 0], 
                [y[:, min_level:max_level, 0] for y in y_pred_list], 
                models, 'Upward Longwave (Vertical Differences)', 
                'Flux Difference [W/m$^2$]', 'Vertical Level')

last_ax = plot_differences(fig, 6, vertical_levels[:-1], 
                         h_true_list[0][:, min_level:max_level-1, 0], 
                         [h[:, min_level:max_level-1, 0] for h in h_pred_list], 
                         models, 'Heating Rates (Longwave, Vertical Differences)', 
                         'Heating Rate Difference [K/day]', 'Vertical Level')

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.suptitle(f"Sample {sample_index}: Vertical Differences Comparison (Levels {min_level}-{max_level-1})", fontsize=16)

# Save the figure
plt.savefig(f'/mydata/deepcloud/yves/Vertical_differences_GNN_HRLU_Models_sample_{sample_index}_levels_{min_level}-{max_level-1}.png', bbox_inches='tight', dpi=300)
print(f"Figure saved as Vertical_differences_GNN_HRLU_Models_sample_{sample_index}_levels_{min_level}-{max_level-1}.png")