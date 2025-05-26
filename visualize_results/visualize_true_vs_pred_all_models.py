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
    #{'name': 'AFNO','path': '/mydata/deepcloud/yves/results-temp/afno_column_1percent_Emb128_clean_tendency_normTarg/test'},
    #{
    #    'name': 'GNN-32-L2',
    #    'path': '/mydata/deepcloud/yves/results-temp/gnn_32_l2_tendency_normTarg/test'
    #},
    #{'name': 'GNN-64-L3', 'path': '/mydata/deepcloud/yves/results-temp/gnn_64_l3_tendency_normTarg/test'},

    # { 'name': 'GNN-128-L2', 'path': '/mydata/deepcloud/yves/results-temp/gnn_128_l2_tendency_normTarg/test'}
    #{'name': 'GNN3d-32-L2', 'path': '/mydata/deepcloud/yves/results-temp/gnn3d_id39_tendency_32_l2/test'},
    #{'name': 'GNN3d-32-L2-Indep','path': '/mydata/deepcloud/yves/results-temp/gnn3d_id39_tendency_32_l2_indep/test'},
    
    
    {'name': 'GNN-3D-64-L2-100', 'path': '/mydata/deepcloud/yves/results-temp/gnn3d_id39_tendency_64_l2_100/test'},
    #{'name': 'GNN-3D-64-L2-100-Indep','path': '/mydata/deepcloud/yves/results-temp/gnn3d_id39_tendency_64_l2_indep_100/test'},
    #{'name': 'GNN-1D-64-L2-100','path': '/mydata/deepcloud/yves/results-temp/gnn_64_l2_tendency_1d_triangle/test'},
    {'name': 'GNN-3D-64-L2-100-NoFully','path': '/mydata/deepcloud/yves/results-temp/gnn3d_id39_tendency_64_l2_100_nofully/test'},

    

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

# Choose samples to visualize
selected_samples = [0, 1, 2]  # Change this to select specific samples

# Define colors and line styles for models
true_color = 'black'
pred_colors = ['blue', 'red', 'green', 'purple']
line_styles = ['-', '--', '-.', ':']

def create_and_save_plot(use_all_samples, sample_idx=None):
    """
    Create and save a plot for either all samples or a specific sample index.
    
    Args:
        use_all_samples: If True, use all samples (average). If False, use specific sample.
        sample_idx: The index of the specific sample to use if use_all_samples is False.
    """
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
        
        # Decide whether to use all samples or only selected one
        if not use_all_samples:
            if sample_idx is not None:
                y_true = y_true[[sample_idx]]
                y_pred = y_pred[[sample_idx]]
                print(f'Using selected sample: {sample_idx}')
        else:
            print(f'Using all {len(y_true)} samples')
        
        # Convert to PyTorch tensors if needed
        if isinstance(y_true, np.ndarray):
            y_true = torch.tensor(y_true)
        if isinstance(y_pred, np.ndarray):
            y_pred = torch.tensor(y_pred)
        
        # Check and handle data dimensions
        print(f'Data dimensions - y_true: {len(y_true.shape)}D, shape: {y_true.shape}')
        
        if len(y_true.shape) == 4:  # 4D: [batch, columns, height, features]
            print(f'Processing 4D data: [batch, columns, height, features]')
            # For 4D data, we need to average over both batch and columns
            if model_idx == 0:
                # For ground truth, take the mean over batch and columns
                y_true_h = torch.mean(y_true, dim=(0, 1))  # (H, F)
                print(f'Averaged ground truth shape: {y_true_h.shape}')
            
            # For predictions, average over batch and columns
            y_pred_h = torch.mean(y_pred, dim=(0, 1))  # (H, F)
            print(f'Averaged predictions shape: {y_pred_h.shape}')
            
        elif len(y_true.shape) == 3:  # 3D: [batch, height, features]
            print(f'Processing 3D data: [batch, height, features]')
            # Check and reshape if necessary for flattened 3D data
            if y_true.shape[-1] == 490:
                y_true = y_true.reshape(-1, 70, 7)  # Reshape to [batch, height, features]
                y_pred = y_pred.reshape(-1, 70, 7)
                print(f'Reshaped to y_true: {y_true.shape}, y_pred: {y_pred.shape}')
            
            # Store the ground truth from the first model only
            if model_idx == 0:
                if use_all_samples:
                    # Take the mean over the batch dimension
                    y_true_h = torch.mean(y_true, dim=0)  # (H, F)
                    print(f'Averaged ground truth across {y_true.shape[0]} samples, shape: {y_true_h.shape}')
                else:
                    # Use the specific sample without averaging
                    y_true_h = y_true[0]  # (H, F)
                    print(f'Single sample ground truth shape: {y_true_h.shape}')
            
            if use_all_samples:
                # For predictions, take mean over batch dimension
                y_pred_h = torch.mean(y_pred, dim=0)  # (H, F)
                print(f'Averaged predictions across {y_pred.shape[0]} samples, shape: {y_pred_h.shape}')
            else:
                # Use the specific sample without averaging
                y_pred_h = y_pred[0]  # (H, F)
                print(f'Single sample prediction shape: {y_pred_h.shape}')
                
        else:
            raise ValueError(f"Unexpected data shape: {y_true.shape}. Expected 3D or 4D tensor.")
        
        # Store the processed prediction
        y_pred_hs.append(y_pred_h)

    # Get height range
    height_range = range(y_true_h.shape[0])

    # Create the consolidated plot with 3x3 grid of subplots
    fig, axes = plt.subplots(3, 3, figsize=(12, 22), sharey=True)
    axes = axes.flatten()  # Flatten to make indexing easier
    
    # Only use the first 7 subplots (since we have 7 targets)
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
        
        # Format subplot - keep title on a single line
        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.7)
        ax.invert_yaxis()
        
        # Add x-label
        ax.set_xlabel(f'[{unit}]', fontsize=12)
        
        # Add y-label to the first subplot of each row
        if i % 3 == 0:
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
    
    # Hide unused subplots
    for i in range(len(target_units), 9):
        axes[i].set_visible(False)

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
    if use_all_samples:
        num_samples = len(y_true) if len(y_true.shape) >= 2 else 1
        sample_desc = f"All Samples (Mean of {num_samples})"
        output_path = f'/mydata/deepcloud/yves/results-temp/true-vs-pred-3d-fully-vs-nonfully-64-all-samples-{num_samples}.png'
    else:
        sample_desc = f"Sample {sample_idx}"
        output_path = f'/mydata/deepcloud/yves/results-temp/true-vs-pred-3d-fully-vs-nonfully-64-sample-{sample_idx}.png'
    
    fig.suptitle(f'True vs. Predicted Values Across All Models - {sample_desc}', fontsize=16, y=0.98)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    # Save the figure
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Visualization saved to: {output_path}")
    plt.close(fig)

# Run the plotting for all samples
create_and_save_plot(use_all_samples=True)

# Run the plotting for individual samples
for sample in selected_samples:
    create_and_save_plot(use_all_samples=False, sample_idx=sample)

print("All visualizations completed.") 