import torch
import pickle
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

models = [
    {
        'name': 'AFNO',
        'path': '/mydata/deepcloud/yves/results-temp/afno_column_1percent_Emb128_clean_tendency_normTarg/test'
    },
    
        # { 'name': 'RF-concat-tendency-norm',        'path': '/mydata/deepcloud/shared/results-temp/afno_column_1percent_Emb128_clean_tendency_RF/test'},
    
    
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

    # >>> choose which test‑set indices you want to look at <<<
    #    – a single sample,    e.g. [17]
    #    – a few samples,      e.g. [3, 8, 12]
    #    – or the first k,     e.g. range(5)
    sample_idx = [134]                 # change this line to taste
    # ---------------------------------------------------------------

    # keep only the requested samples
    y_true = y_true[sample_idx]
    y_pred = y_pred[sample_idx]
    # ---------------------------------------------------------------

    # Convert y_true and y_pred to PyTorch tensors if they are numpy arrays
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred)
    
    # Check and reshape if necessary
    if y_true.shape[-1] == 490:  # Flattened shape detected
        y_true = y_true.reshape(-1, 70, 7)  # Reshape to [batch, height, features]
        y_pred = y_pred.reshape(-1, 70, 7)
        print(f'Reshaped y_true: {y_true.shape}')

        print(f'Reshaped y_pred: {y_pred.shape}')

    y_mae_h = torch.mean(torch.abs(y_true - y_pred), dim=0)
    # y_mae_h should now have shape [height, 7] (or [7, height], check accordingly).

    # Optionally flip the vertical dimension so that index=0 corresponds to the top
    # y_mae_h = torch.flip(y_mae_h, dims=[0])
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
                title=None, mask=None, train_target_means=None):
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
        ylabel='Height index' if i in [0, 3, 6] else None,
        xlabel=f'MAE [{units}]',
        mask=mask,
    )
    ax_list.append(ax)
    
    
# Legend on the first subplot
ax_list[0].legend(fontsize=12, loc='best')

plt.tight_layout()
plt.savefig('/mydata/deepcloud/yves/Tendency-normTarget-mae-afno-singleSample.png',
            bbox_inches='tight', dpi=300)   
plt.show()