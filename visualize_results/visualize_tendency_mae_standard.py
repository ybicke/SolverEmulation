import torch
import pickle
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import os

models = [
    #{'name': 'GNN-3D-64-L2-100', 'path': '/mydata/deepcloud/yves/results-temp/gnn3d_id39_tendency_64_l2_100/test'},
    #{'name': 'GNN-3D-64-L2-100-new', 'path': '/mydata/deepcloud/yves/results-temp/gnn3d_id39_tendency_64_l2_100_new/test'},
    #{'name': 'GNN-3D-64-L2-100-Indep','path': '/mydata/deepcloud/yves/results-temp/gnn3d_id39_tendency_64_l2_indep_100/test'},
    #{'name': 'GNN-64-l2-100eps-noFully','path': '/mydata/deepcloud/yves/results-temp/gnn3d_id39_tendency_64_l2_100_nofully/test'},
    #{'name': 'GNN-1D-64-L2-100','path': '/mydata/deepcloud/yves/results-temp/gnn_64_l2_tendency_1d_triangle/test'},


    #{'name': 'GNN-3D-64-L2-100-Indep','path': '/mydata/deepcloud/yves/results-temp/gnn3d_id39_tendency_64_l2_indep_100/test'},
    {'name': 'GNN-3D-64-L2-100','path': '/mydata/deepcloud/yves/results-temp/gnn3d_id39_tendency_64_l2_100_new/test'},
    
    #{'name': 'AFNO-1D-128','path': '/mydata/deepcloud/yves/results-temp/afno_column_1percent_Emb128_clean_tendency_normTarg/test'},
    #{'name': 'GNN-1D-64-L2-100','path': '/mydata/deepcloud/yves/results-temp/gnn_64_l2_tendency_1d_triangle_fully_connected/test'},
    
    {'name': 'GT-3D-64-L4-K2-Drop03-100','path': '/mydata/deepcloud/yves/results-temp/graph_transformer_hybrid_3d_64_l4_k2_drop03_simplified_100/test'},
    
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
    - 'ys' is a list of y-values (MAEs), one for each model, each shaped (70,).
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
height_range = range(y_mae_hs[0].shape[0])

# Create MAE figure
fig_mae = plt.figure(figsize=(12, 18))
ax_list_mae = []

for i, (label, units) in enumerate(target_units.items()):
    subplot_idx = (3, 3, i+1)
    channel_data = [y[:, i] for y in y_mae_hs]
    ax = add_supplot(
        fig_mae, 
        x=height_range,
        ys=channel_data,
        id=subplot_idx,
        models_name=models_name,
        title=label,
        ylabel='Height index' if i in [0, 3, 6] else None,
        xlabel=f'MAE [{units}]',
        mask=mask,
    )
    ax_list_mae.append(ax)

# Legend on the first subplot
ax_list_mae[0].legend(fontsize=12, loc='best')

plt.tight_layout()
plt.savefig('/mydata/deepcloud/yves/final_results/3D_Tendency_GNN_vs_GT_MAE.png',
            bbox_inches='tight', dpi=300)

plt.show()