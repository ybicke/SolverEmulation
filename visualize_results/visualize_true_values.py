import torch
import pickle
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
from itertools import cycle

# Define the model name
model_name = 'gnn_graphCast_Emb512_plateau_00005' 
model_path = f'/mydata/deepcloud/yves/results_git/{model_name}/test'

# Define the model dictionary
models = [{'name': model_name, 'path': model_path}]

y_true_list, y_pred_list, h_true_list, h_pred_list = [], [], [], []

for model in models:
    test_path = model['path']
    print(f'loading test files... ({test_path})')
    with open(join(test_path, 'y_true.pickle'), 'rb') as handle:
        y_true = pickle.load(handle)
    with open(join(test_path, 'y_pred.pickle'), 'rb') as handle:
        y_pred = pickle.load(handle)
    with open(join(test_path, 'h_true.pickle'), 'rb') as handle:
        h_true = pickle.load(handle)
    with open(join(test_path, 'h_pred.pickle'), 'rb') as handle:
        h_pred = pickle.load(handle)

    y_true_list.append(y_true)
    y_pred_list.append(y_pred)
    h_true_list.append(h_true)
    h_pred_list.append(h_pred)

def add_subplot(fig, x, y_trues, y_preds, id, models, xlabel=None, ylabel=None, title=None, flux=True, subset_slice=slice(None)):
    ax = fig.add_subplot(*id)
    color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    for y_true, y_pred, model in zip(y_trues, y_preds, models):
        color = next(color_cycle)
        # Check the dimensionality and adjust accordingly
        if y_true.ndim == 3:
            y_true_mean = y_true[:, subset_slice, :].mean(axis=(0, 1))
            y_pred_mean = y_pred[:, subset_slice, :].mean(axis=(0, 1))
        elif y_true.ndim == 2:
            y_true_mean = y_true[subset_slice, :].mean(axis=0)
            y_pred_mean = y_pred[subset_slice, :].mean(axis=0)
        else:
            raise ValueError("Unexpected number of dimensions in data tensor")

        ax.plot(y_true_mean, x, label='True Values', linestyle='-', color=color)
        ax.plot(y_pred_mean, x, label='Predicted Values', linestyle='--', color=color)
    ax.grid()
    ax.set_xlabel(xlabel if xlabel else '')
    ax.set_ylabel(ylabel if ylabel else '')
    ax.set_title(title if title else '')
    ax.legend(fontsize="8", loc='upper right')  # Add legend at the top right with smaller font size
    return ax

fig = plt.figure(figsize=(12, 13))  # Increase the figure width

# Define the subset slice, e.g., averaging over the first 50 samples
subset = slice(130, 131)

# Flux and heating rate plots with adjusted slicing
add_subplot(fig, x=range(71), y_trues=[y[:, :, 3] for y in y_true_list], y_preds=[y[:, :, 3] for y in y_pred_list], id=(2, 3, 1), models=models, title='Downward Shortwave Flux', xlabel='Flux [W/m$^2$]', ylabel='Vertical Level', subset_slice=subset)
add_subplot(fig, x=range(71), y_trues=[y[:, :, 1] for y in y_true_list], y_preds=[y[:, :, 1] for y in y_pred_list], id=(2, 3, 4), models=models, title='Downward Longwave Flux', xlabel='Flux [W/m$^2$]', ylabel='Vertical Level', subset_slice=subset)
add_subplot(fig, x=range(71), y_trues=[y[:, :, 2] for y in y_true_list], y_preds=[y[:, :, 2] for y in y_pred_list], id=(2, 3, 2), models=models, title='Upward Shortwave Flux', xlabel='Flux [W/m$^2$]', ylabel='Vertical Level', subset_slice=subset)
add_subplot(fig, x=range(71), y_trues=[y[:, :, 0] for y in y_true_list], y_preds=[y[:, :, 0] for y in y_pred_list], id=(2, 3, 5), models=models, title='Upward Longwave Flux', xlabel='Flux [W/m$^2$]', ylabel='Vertical Level', subset_slice=subset)

# Heating rate plots
add_subplot(fig, x=range(70), y_trues=[h[:, :, 1] for h in h_true_list], y_preds=[h[:, :, 1] for h in h_pred_list], id=(2, 3, 3), models=models, title='Heating Rates (Shortwave)', xlabel='Heating Rate [K/day]', ylabel='Vertical Level', flux=False, subset_slice=subset)
add_subplot(fig, x=range(70), y_trues=[h[:, :, 0] for h in h_true_list], y_preds=[h[:, :, 0] for h in h_pred_list], id=(2, 3, 6), models=models, title='Heating Rates (Longwave)', xlabel='Heating Rate [K/day]', ylabel='Vertical Level', flux=False, subset_slice=subset)

# Create a single title for the entire plot
fig.suptitle(f'{model_name} - True vs Predicted Values (Sample {subset.start})', fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the suptitle
plt.savefig(f'/mydata/deepcloud/yves/results_git/{model_name}_real_values.png', bbox_inches='tight', dpi=300)