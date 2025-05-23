import torch
import pickle
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

models = [
    #{
    #    'name': 'AFNO',
    #    'path': '/mydata/deepcloud/yves/results-temp/afno_column_1percent_Emb128_clean_tendency_normTarg/test'
    #},
    #{
    #    'name': 'GNN-32-L2',
    #    'path': '/mydata/deepcloud/yves/results-temp/gnn_32_l2_tendency_normTarg/test'
    #},
    #{
    #    'name': 'GNN-64-L3',
    #    'path': '/mydata/deepcloud/yves/results-temp/gnn_64_l3_tendency_normTarg/test'
    #},
    #{
    #    'name': 'GNN-128-L2',
    #    'path': '/mydata/deepcloud/yves/results-temp/gnn_128_l2_tendency_normTarg/test'
    #},
    
    {'name': 'GNN-32-L2', 'path': '/mydata/deepcloud/yves/results-temp/gnn3d_id39_tendency_32_l2/test'},
    {'name': 'GNN-32-L2-Indep','path': '/mydata/deepcloud/yves/results-temp/gnn3d_id39_tendency_32_l2_indep/test'},
]

mae_values = []
# Store statistics for reference
stats = {
    'global_min': [],
    'global_max': [],
    'global_mean': [],
    'global_std': []
}

for model in models:
    test_path = model['path']
    print(f'Loading test files... ({test_path})')

    with open(join(test_path, 'y_true.pickle'), 'rb') as handle:
        y_true = pickle.load(handle)
    print(f'y_true shape: {y_true.shape}')

    with open(join(test_path, 'y_pred.pickle'), 'rb') as handle:
        y_pred = pickle.load(handle)
    print(f'y_pred shape: {y_pred.shape}')
    
    # Convert y_true and y_pred to PyTorch tensors if they are numpy arrays
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred)
    
    # Check data dimensions and handle appropriately
    print(f'Data dimensions - y_true: {len(y_true.shape)}D, shape: {y_true.shape}')
    
    if len(y_true.shape) == 4:  # 4D: [batch, columns, height, features]
        print(f'Processing 4D data: [batch, columns, height, features]')
        # Calculate absolute error first (keeping all dimensions)
        abs_error = torch.abs(y_true - y_pred)
        # Average over both batch and column dimensions
        mae_by_height = torch.mean(abs_error, dim=(0, 1))  # Result: [height, 7]
        
        # Calculate statistics on flattened data
        # Reshape to combine [batch, columns, height] into one dimension, keeping features separate
        y_true_reshaped = y_true.reshape(-1, y_true.shape[-1])
        
    elif len(y_true.shape) == 3:  # 3D: [batch, height, features]
        print(f'Processing 3D data: [batch, height, features]')
        # Check and reshape if necessary for flattened 3D data
        if y_true.shape[-1] == 490:  # Flattened shape detected
            y_true = y_true.reshape(-1, 70, 7)  # Reshape to [batch, height, features]
            y_pred = y_pred.reshape(-1, 70, 7)
            print(f'Reshaped to y_true: {y_true.shape}, y_pred: {y_pred.shape}')
        
        # Calculate absolute error (MAE)
        abs_error = torch.abs(y_true - y_pred)
        # Calculate mean error across all samples (but keep height dimension)
        mae_by_height = torch.mean(abs_error, dim=0)  # Shape: [height, 7]
        
        # Reshape for statistics calculation
        y_true_reshaped = y_true.reshape(-1, y_true.shape[-1])
        
    else:
        raise ValueError(f"Unexpected data shape: {y_true.shape}. Expected 3D or 4D tensor.")
    
    # Calculate and store statistics (same for both 3D and 4D)
    stats['global_min'].append(torch.min(y_true_reshaped, dim=0)[0])
    stats['global_max'].append(torch.max(y_true_reshaped, dim=0)[0])
    stats['global_mean'].append(torch.mean(y_true_reshaped, dim=0))
    stats['global_std'].append(torch.std(y_true_reshaped, dim=0))
    
    # Add the calculated MAE values
    mae_values.append(mae_by_height)
    print(f'MAE shape: {mae_by_height.shape}')

# Print statistics for reference
for i, model in enumerate(models):
    print(f"\nStatistics for {model['name']}:")
    for stat_name, stat_values in stats.items():
        print(f"  {stat_name}: {stat_values[i].tolist()}")

target_units = {
    "Sum of Temperature Tendency": "K s-1", 
    "Dynamical Temperature Tendency": "K s-1",
    "Sum of Zonal Wind Tendency": "m s-2",
    "Sum of Meridional Wind Tendency": "m s-2",
    "Convective Tend. Absolute Humidity": "kg m-3 s-1",
    "Convective Tend. Cloud Water Mass Density": "kg m-3 s-1",
    "Convective Tend. Cloud Ice Mass Density": "kg m-3 s-1"
}

def add_supplot(fig, x, ys, id, models_name, xlabel=None, ylabel=None, 
                title=None, mask=None):
    """
    Helper function to add a subplot to 'fig'.
    - 'x' is the array for the vertical axis (e.g. range(70)).
    - 'ys' is a list of y-values (MAE values), one for each model, each shaped (70,).
    - 'id' is a tuple (nrows, ncols, index) for subplot placement.
    - 'mask' is a list of booleans indicating which models to plot.
    """
    ax = fig.add_subplot(*id)
    for y, model_name, msk in zip(ys, models_name, mask):
        if msk:
            ax.plot(y, x, label=f'{model_name}')

    ax.grid(True)
    ax.invert_yaxis()
    # Increase tick label size
    ax.tick_params(axis='both', which='major', labelsize=12)
    if title:
        ax.set_title(title if title else '', fontsize=13)  # Increase title size
    if ylabel:
        ax.set_ylabel(ylabel if ylabel else '', fontsize=14)  # Increase label size
    if xlabel:
        ax.set_xlabel(xlabel if xlabel else '', fontsize=14)  # Increase label size

    return ax

# Prepare the figure
models_name = [model['name'] for model in models]
mask = [True] * len(models_name)

# If mae_values[0] has shape (70, 7), the first dimension is height=70.
height_range = range(mae_values[0].shape[0])      

fig = plt.figure(figsize=(12, 18))

# We create one subplot for each of the 7 channels
# We'll arrange them in a 3x3 grid, so 9 total slots (2 will remain unused)
ax_list = []

for i, (label, units) in enumerate(target_units.items()):
    # Subplot index in a 3x3 grid is i+1
    # (3 rows, 3 columns, i+1)
    subplot_idx = (3, 3, i+1)

    # Collect the y-values for each model at channel i
    channel_data = [y[:, i] for y in mae_values]
    
    ax = add_supplot(
        fig, 
        x=height_range,
        ys=channel_data,
        id=subplot_idx,
        models_name=models_name,
        title=label,
        ylabel='Height index' if i in [0, 3, 6] else None,
        xlabel=f'MAE {units}',
        mask=mask,
    )
    ax_list.append(ax)
    
    
# Legend on the first subplot
ax_list[0].legend(fontsize=12, loc='best')

plt.tight_layout()
plt.savefig('/mydata/deepcloud/yves/results-temp/tendency-mae-3d-indep-vs-horiz-32.png',
            bbox_inches='tight', dpi=300)   
plt.show()