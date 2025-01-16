import torch
import pickle
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

models = [
    {
        'name': 'AFNO-Emb128-concat-tendency-norm',
        'path': '/mydata/deepcloud/shared/results-temp/afno_column_1percent_Emb128_clean_tendency_normTarg/test'
    },
]

y_mae_hs = []
train_target_means = []

for model in models:
    test_path = model['path']
    print(f'Loading test files... ({test_path})')

    with open(join(test_path, 'y_true.pickle'), 'rb') as handle:
        y_true = pickle.load(handle)
    print(f'y_true shape: {y_true.shape}')

    with open(join(test_path, 'y_pred.pickle'), 'rb') as handle:
        y_pred = pickle.load(handle)
    print(f'y_pred shape: {y_pred.shape}')

    with open(join(test_path, 'train_target_mean_first.pickle'), 'rb') as handle:
        train_target_mean = pickle.load(handle)
    train_target_means.append(train_target_mean)

    # Calculate mean absolute error along the batch/time dimension(s).
    # Adjust 'dim' depending on your shapes (e.g. [batch, time, height, 7] -> dim=[0,1])
    if len(y_true.shape) == 7:
        dim = [0, 1]
    else:
        dim = 0

    y_mae_h = torch.mean(torch.abs(y_true - y_pred), dim=dim)
    # y_mae_h should now have shape [height, 7] (or [7, height], check accordingly).

    # Optionally flip the vertical dimension so that index=0 corresponds to the top
    # y_mae_h = torch.flip(y_mae_h, dims=[0])

    y_mae_hs.append(y_mae_h)
    
    
    # Calculate MSE metrics
    baseline_mse = torch.mean((y_true - train_target_mean)**2)
    model_mse = torch.mean((y_true - y_pred)**2)
    print(f'Baseline MSE: {baseline_mse:.15f}')
    print(f'Model MSE: {model_mse:.15f}')
    print(f'MSE Ratio (Model / Baseline): {model_mse / baseline_mse:.15f}')
    

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
                title=None, log_scale=True, mask=None, train_target_means=None):
    """
    Helper function to add a subplot to 'fig'.
    - 'x' is the array for the vertical axis (e.g. range(70)).
    - 'ys' is a list of y-values (MAEs), one for each model, each shaped (70,).
    - 'id' is a tuple (nrows, ncols, index) for subplot placement.
    - 'mask' is a list of booleans indicating which models to plot.
    - 'train_target_means' is a list of mean values for each model.
    """
    ax = fig.add_subplot(*id)
    for y, model_name, msk in zip(ys, models_name, mask):
        if msk:
            ax.plot(y, x, label=f'{model_name}')
    ax.grid()
    ax.invert_yaxis()
    ax.set_ylabel(ylabel if ylabel else '')
    ax.set_title(title if title else '')
    if log_scale:
        ax.set_xscale('log')
        
        # Calculate the range of the data
        x_min = min(min(y) for y in ys if len(y) > 0)
        x_max = max(max(y) for y in ys if len(y) > 0)
        
        # Clear existing ticks
        ax.xaxis.clear()
        
        # Set the tick positions dynamically based on the data range
        num_ticks = 5
        tick_positions = np.logspace(np.log10(x_min), np.log10(x_max), num_ticks)
        ax.set_xticks(tick_positions)
        
        # Set the tick labels to use scientific notation and round to 2 decimal places
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.2e}'))
        
    # Set the x-axis label after updating the tick positions and labels
    ax.set_xlabel(xlabel if xlabel else '')
        
    return ax

# Prepare the figure
models_name = [model['name'] for model in models]
mask = [True] * len(models_name)

# If y_mae_hs[0] has shape (70, 7), the first dimension is height=70.
height_range = range(y_mae_hs[0].shape[0])      

fig = plt.figure(figsize=(12, 18))

# We create one subplot for each of the 7 channels
# We'll arrange them in a 3x3 grid, so 9 total slots (2 will remain unused)
# Feel free to choose a layout that suits you.
ax_list = []

for i, (label, units) in enumerate(target_units.items()):
    # Subplot index in a 3x3 grid is i+1
    # (3 rows, 3 columns, i+1)
    subplot_idx = (3, 3, i+1)

    # Collect the y-values for each model at channel i
    # (assuming shape is [height, 7], so y_mae_hs[m][:, i] is the i-th channel for model m)
    channel_data = [y[:, i] for y in y_mae_hs]
    
    ax = add_supplot(
        fig, 
        x=height_range,
        ys=channel_data,
        id=subplot_idx,
        models_name=models_name,
        title=label,
        ylabel='Height index' if i in [0, 3, 6] else None,  # just an example for labeling
        xlabel=f'MAE [{units}]',
        mask=mask,
        # train_target_means=[mean[:, i] for mean in train_target_means]
    )
    ax_list.append(ax)
    
    
# Add legend to the first subplot (or any axis of your choice)
ax_list[0].legend(fontsize="10", loc='best')

plt.tight_layout()
plt.savefig('/mydata/deepcloud/shared/results-temp/tendency-data-first-norm-targetMean.png',
            bbox_inches='tight', dpi=300)   
plt.show()