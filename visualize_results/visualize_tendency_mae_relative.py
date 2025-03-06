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
    # Add more models here if needed
]

y_norm_errors = []
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

    # Calculate absolute error
    abs_error = torch.abs(y_true - y_pred)
    
    # Reshape y_true to combine batch and height dimensions for finding global min/max per feature
    y_true_reshaped = y_true.reshape(-1, 7)  # shape: [batch*height, 7]
    
    # Calculate min-max range for each feature across all samples and heights
    min_vals, _ = torch.min(y_true_reshaped, dim=0)  # shape: [7]
    max_vals, _ = torch.max(y_true_reshaped, dim=0)  # shape: [7]
    value_range = max_vals - min_vals  # shape: [7]
    
    
    # Normalize the error by the range (broadcasting the range across all samples and heights)
    normalized_error = abs_error / value_range.unsqueeze(0).unsqueeze(0)
    
    # Calculate mean normalized error across all samples
    y_norm_error_h = torch.mean(normalized_error, dim=0)
    # y_norm_error_h should now have shape [height, 7]
    
    y_norm_errors.append(y_norm_error_h)

target_units = {
    "Sum of Temperature Tendency": "normalized error", 
    "Dynamical Temperature Tendency": "normalized error",
    "Sum of Zonal Wind Tendency": "normalized error",
    "Sum of Meridional Wind Tendency": "normalized error",
    "Convective Tend. Absolute Humidity": "normalized error",
    "Convective Tend. Cloud Water Mass Density": "normalized error",
    "Convective Tend. Cloud Ice Mass Density": "normalized error"
}

def add_supplot(fig, x, ys, id, models_name, xlabel=None, ylabel=None, 
                title=None, mask=None, train_target_means=None):
    """
    Helper function to add a subplot to 'fig'.
    - 'x' is the array for the vertical axis (e.g. range(70)).
    - 'ys' is a list of y-values (normalized errors), one for each model, each shaped (70,).
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

# If y_norm_errors[0] has shape (70, 7), the first dimension is height=70.
height_range = range(y_norm_errors[0].shape[0])      

fig = plt.figure(figsize=(12, 18))

# We create one subplot for each of the 7 channels
# We'll arrange them in a 3x3 grid, so 9 total slots (2 will remain unused)
ax_list = []

for i, (label, units) in enumerate(target_units.items()):
    # Subplot index in a 3x3 grid is i+1
    # (3 rows, 3 columns, i+1)
    subplot_idx = (3, 3, i+1)

    # Collect the y-values for each model at channel i
    channel_data = [y[:, i] for y in y_norm_errors]
    
    ax = add_supplot(
        fig, 
        x=height_range,
        ys=channel_data,
        id=subplot_idx,
        models_name=models_name,
        title=label,
        ylabel='Height index' if i in [0, 3, 6] else None,
        xlabel=f'Normalized Error (MAE/Global Range)',
        mask=mask,
    )
    ax_list.append(ax)
    
    
# Legend on the first subplot
ax_list[0].legend(fontsize=12, loc='best')

plt.tight_layout()
plt.savefig('/mydata/deepcloud/yves/results-temp/tendency-norm-targetMean-normalized-error-afno.png',
            bbox_inches='tight', dpi=300)   
plt.show()