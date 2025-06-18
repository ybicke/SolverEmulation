import torch
import pickle
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import os

models = [
    {'name': 'GNN-3D-64-L2-100', 'path': '/mydata/deepcloud/yves/results-temp/gnn3d_id39_tendency_64_l2_100/test'},
    #{'name': 'GNN-3D-64-L2-100-Indep','path': '/mydata/deepcloud/yves/results-temp/gnn3d_id39_tendency_64_l2_indep_100/test'},
    #{'name': 'GNN-1D-64-L2-100','path': '/mydata/deepcloud/yves/results-temp/gnn_64_l2_tendency_1d_triangle/test'},
    {'name': 'GNN-64-l2-100eps-noFully','path': '/mydata/deepcloud/yves/results-temp/gnn3d_id39_tendency_64_l2_100_nofully/test'},
]

y_r2_hs = []

def r2_score(y_true, y_pred, y_mean=None, eps=1e-10):
    """Calculate R² score between true and predicted values with better handling of edge cases.
    R² = 1 - (Σ(y_true - y_pred)²) / (Σ(y_true - y_mean)²)
    """
    if y_mean is None:
        y_mean = torch.mean(y_true)
    
    # Get the scale of the true values
    y_scale = torch.std(y_true)
    if y_scale == 0:
        y_scale = torch.mean(torch.abs(y_true))
    if y_scale == 0:
        y_scale = 1.0

    # Normalize by the scale to handle different magnitudes
    y_true_norm = y_true / y_scale
    y_pred_norm = y_pred / y_scale
    y_mean_norm = y_mean / y_scale
    
    # Residual sum of squares
    ss_res = torch.sum((y_true_norm - y_pred_norm) ** 2)
    # Total sum of squares
    ss_tot = torch.sum((y_true_norm - y_mean_norm) ** 2)
    
    # Handle the case where variance is very small
    if ss_tot < eps:
        # For near-constant data, check if predictions are close
        if ss_res < eps:
            return torch.tensor(1.0)  # Perfect prediction for constant data
        else:
            # Return NMSE (Normalized Mean Squared Error) transformed to R² scale
            return torch.tensor(1.0 - ss_res / len(y_true_norm))
    
    r2 = 1 - (ss_res / ss_tot)
    return r2

def normalized_rmse(y_true, y_pred):
    """Calculate normalized RMSE as an alternative metric"""
    # Get the scale of the true values
    y_scale = torch.std(y_true)
    if y_scale == 0:
        y_scale = torch.mean(torch.abs(y_true))
    if y_scale == 0:
        y_scale = 1.0
    
    # Calculate RMSE and normalize
    rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2))
    return rmse / y_scale

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
        # Calculate or load means
        means_file = join(test_path, 'y_means.pickle')
        if not os.path.exists(means_file):
            y_means = torch.zeros((y_true.shape[2], y_true.shape[3]))
            for h in range(y_true.shape[2]):
                for f in range(y_true.shape[3]):
                    y_means[h, f] = torch.mean(y_true[..., h, f])
            with open(means_file, 'wb') as handle:
                pickle.dump(y_means, handle)
        else:
            with open(means_file, 'rb') as handle:
                y_means = pickle.load(handle)
        
        # Calculate R² and normalized RMSE
        y_r2_h = torch.zeros((y_true.shape[2], y_true.shape[3]))
        y_nrmse_h = torch.zeros((y_true.shape[2], y_true.shape[3]))
        zero_var_heights = []
        
        for h in range(y_true.shape[2]):
            for f in range(y_true.shape[3]):
                y_true_flat = y_true[..., h, f].flatten()
                y_pred_flat = y_pred[..., h, f].flatten()
                
                if torch.std(y_true_flat) == 0:
                    zero_var_heights.append((h, f))
                
                y_r2_h[h, f] = r2_score(y_true_flat, y_pred_flat, y_means[h, f])
                y_nrmse_h[h, f] = normalized_rmse(y_true_flat, y_pred_flat)
        
        if zero_var_heights:
            print(f"Found {len(zero_var_heights)} constant (zero variance) features")
    
    else:  # Shape: [batch, height, features]
        # Calculate or load means
        means_file = join(test_path, 'y_means.pickle')
        if not os.path.exists(means_file):
            y_means = torch.zeros((y_true.shape[1], y_true.shape[2]))
            for h in range(y_true.shape[1]):
                for f in range(y_true.shape[2]):
                    y_means[h, f] = torch.mean(y_true[:, h, f])
            with open(means_file, 'wb') as handle:
                pickle.dump(y_means, handle)
        else:
            with open(means_file, 'rb') as handle:
                y_means = pickle.load(handle)
        
        # Calculate R² and normalized RMSE
        y_r2_h = torch.zeros((y_true.shape[1], y_true.shape[2]))
        y_nrmse_h = torch.zeros((y_true.shape[1], y_true.shape[2]))
        zero_var_heights = []
        
        for h in range(y_true.shape[1]):
            for f in range(y_true.shape[2]):
                y_true_flat = y_true[:, h, f]
                y_pred_flat = y_pred[:, h, f]
                
                if torch.std(y_true_flat) == 0:
                    zero_var_heights.append((h, f))
                
                y_r2_h[h, f] = r2_score(y_true_flat, y_pred_flat, y_means[h, f])
                y_nrmse_h[h, f] = normalized_rmse(y_true_flat, y_pred_flat)
        
        if zero_var_heights:
            print(f"Found {len(zero_var_heights)} constant (zero variance) features")
    
    y_r2_hs.append(y_r2_h)

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
    - 'ys' is a list of y-values (R²), one for each model, each shaped (70,).
    - 'id' is a tuple (nrows, ncols, index) for subplot placement.
    - 'mask' is a list of booleans indicating which models to plot.
    """
    ax = fig.add_subplot(*id)
    for y, model_name, msk in zip(ys, models_name, mask):
        if msk:
            ax.plot(y, x, label=f'{model_name}')

    ax.grid(True)
    ax.invert_yaxis()
    ax.tick_params(axis='both', which='major', labelsize=12)
    if title:
        ax.set_title(title if title else '', fontsize=13)
    if ylabel:
        ax.set_ylabel(ylabel if ylabel else '', fontsize=14)
    if xlabel:
        ax.set_xlabel(xlabel if xlabel else '', fontsize=14)

    return ax

# Prepare data
models_name = [model['name'] for model in models]
mask = [True] * len(models_name)
height_range = range(y_r2_hs[0].shape[0])

# Create R² figure
fig_r2 = plt.figure(figsize=(12, 18))
ax_list_r2 = []

for i, (label, _) in enumerate(target_units.items()):
    subplot_idx = (3, 3, i+1)
    channel_data = [y[:, i] for y in y_r2_hs]
    ax = add_supplot(
        fig_r2, 
        x=height_range,
        ys=channel_data,
        id=subplot_idx,
        models_name=models_name,
        title=f'{label} - R²',
        ylabel='Height index' if i in [0, 3, 6] else None,
        xlabel='R² Score',
        mask=mask,
    )
    ax_list_r2.append(ax)

# Legend on the first subplot
ax_list_r2[0].legend(fontsize=12, loc='best')

plt.tight_layout()
plt.savefig('/mydata/deepcloud/yves/3D-Tendency-64-nonFully-vs-horizontal-r2.png',
            bbox_inches='tight', dpi=300)

plt.show() 