import torch
import pickle
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
from itertools import cycle

# Define the model names and paths
model_names = [
    
    #'afno_column_1percent_Emb128_afno_crossAttention_clean',
    #'afno_column_1percent_Emb128_afno_crossAttention_clean1',
    
    #'afno_column_1percent_Emb128_easyConcat_clean_smooth33',
    #'afno_column_1percent_Emb128_easyConcat_clean_smooth34',
    #'afno_column_1percent_Emb128_easyConcat_clean_smooth35',
    
    #'afno_column_1percent_Emb128_easyConcat_004',
    #'afno_column_1percent_Emb128_easyConcat_008',
    #'afno_column_1percent_Emb128_easyConcat_012'
    
    #'afno_column_1percent_Emb128_easyConcat_00007',
    #'afno_column_1percent_Emb128_easyConcat_00014',
    #'afno_column_1percent_Emb128_easyConcat_00036',
    #'afno_column_1percent_Emb128_easyConcat_00071',

    
    #'afno_column_1percent_Emb128_easyConcat_clean_smooth31_new',
    #'afno_column_1percent_Emb128_easyConcat_clean_smooth32_new',
    #'afno_column_1percent_Emb128_easyConcat_clean_smooth30'
    
    #'afno_column_1percent_Emb128_afno_crossAttention_clean_2dnorm_new',
    #'afno_column_1percent_Emb128_afno_crossAttention_clean_expand_new',
    #'afno_column_1percent_Emb128_afno_crossAttention_clean_expandPos_new'
    
    'afno_column_1percent_Emb128_clean',
    #vit_column_1percent_emb128_h4',
    #'vit_column_1percent_Emb128_HeightSpecificSigmoid_diffNorm',
    'afno_column_1percent_Emb128_HeightSpecificSigmoid_diffNorm',
    # 'afno_column_1percent_Emb128_HeightSpecificSigmoid_lwup'


]

model_paths = [
    f'/mydata/deepcloud/yves/results_git/{model_names[0]}/test',
    f'/mydata/deepcloud/yves/results_git/{model_names[1]}/test',
    #f'/mydata/deepcloud/yves/results_git/{model_names[2]}/test',
    #f'/mydata/deepcloud/yves/results_git/{model_names[3]}/test',
    #f'/mydata/deepcloud/yves/results_git/{model_names[4]}/test',
    #f'/mydata/deepcloud/yves/results_git/{model_names[5]}/test'

]

models = [{'name': name, 'path': path} for name, path in zip(model_names, model_paths)]

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

    def add_subplot(fig, x, y_trues, y_preds, id, models, xlabel=None, ylabel=None, title=None, subset_slice=slice(None)):
        ax = fig.add_subplot(*id)
        color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        for i, (y_true, y_pred, model) in enumerate(zip(y_trues, y_preds, models)):
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

            # Compute differences between neighboring vertical levels
            delta_y_true = np.diff(y_true_mean)
            delta_y_pred = np.diff(y_pred_mean)

            # Adjust x accordingly
            x_array = np.array(x)
            x_diff = x_array[:-1]

            # Plot true values with different line styles for each model
            if i == 0:
                ax.plot(delta_y_true, x_diff, linestyle='-', color='red', linewidth=1.5, label='True')
            else:
                ax.plot(delta_y_true, x_diff, linestyle='--', color='black', linewidth=1.5)

            # Plot predicted values
            ax.plot(delta_y_pred, x_diff, label=f'{model["name"]}', linestyle='--', color=color)

        ax.grid()
        ax.set_xlabel(xlabel if xlabel else '')
        ax.set_ylabel(ylabel if ylabel else '')
        ax.set_title(title if title else '')
        return ax

# Increase the figure width to make the plots wider
fig = plt.figure(figsize=(20, 13))

# Define the subset slice, e.g., averaging over the first 5 samples
subset = slice(49, 50)

# Flux and heating rate plots with adjusted slicing
add_subplot(fig, x=range(35, 71), y_trues=[y[:, 35:, 3] for y in y_true_list], y_preds=[y[:, 35:, 3] for y in y_pred_list], id=(2, 3, 1), models=models, title='Downward Shortwave', xlabel='Flux Difference [W/m$^2$]', ylabel='Vertical Level', subset_slice=subset)
add_subplot(fig, x=range(35, 71), y_trues=[y[:, 35:, 1] for y in y_true_list], y_preds=[y[:, 35:, 1] for y in y_pred_list], id=(2, 3, 4), models=models, title='Downward Longwave', xlabel='Flux Difference [W/m$^2$]', ylabel='Vertical Level', subset_slice=subset)
add_subplot(fig, x=range(35, 71), y_trues=[y[:, 35:, 2] for y in y_true_list], y_preds=[y[:, 35:, 2] for y in y_pred_list], id=(2, 3, 2), models=models, title='Upward Shortwave', xlabel='Flux Difference [W/m$^2$]', ylabel='Vertical Level', subset_slice=subset)
add_subplot(fig, x=range(35, 71), y_trues=[y[:, 35:, 0] for y in y_true_list], y_preds=[y[:, 35:, 0] for y in y_pred_list], id=(2, 3, 5), models=models, title='Upward Longwave', xlabel='Flux Difference [W/m$^2$]', ylabel='Vertical Level', subset_slice=subset)

# Heating rate plots
add_subplot(fig, x=range(35, 70), y_trues=[h[:, 35:, 1] for h in h_true_list], y_preds=[h[:, 35:, 1] for h in h_pred_list], id=(2, 3, 3), models=models, title='Heating Rates (Shortwave)', xlabel='MAE Difference [K/day]', ylabel='Vertical Level', subset_slice=subset)
last_ax = add_subplot(fig, x=range(35, 70), y_trues=[h[:, 35:, 0] for h in h_true_list], y_preds=[h[:, 35:, 0] for h in h_pred_list], id=(2, 3, 6), models=models, title='Heating Rates (Longwave)', xlabel='MAE Difference [K/day]', ylabel='Vertical Level', subset_slice=subset)

# Create a single title for the entire plot
# fig.suptitle(f'Comparison of {", ".join([model["name"] for model in models])}', fontsize=18)

# Create a single legend at the bottom center of the plot
handles, labels = last_ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(models)+1, fontsize="12")

# Adjust the spacing between subplots to make room for the legend
plt.tight_layout(rect=[0, 0.05, 1, 0.96])  # Adjust the bottom spacing as needed

# plt.savefig(f'/mydata/deepcloud/yves/results_git/easyConcat_clean_sparse2_difference_50.png', bbox_inches='tight', dpi=300)#
# plt.savefig(f'/mydata/deepcloud/yves/results_git/easyConcat_smooth_difference_high_50.png', bbox_inches='tight', dpi=300)
# plt.savefig(f'/mydata/deepcloud/yves/results_git/easyConcat_smooth_difference_percentile_50.png', bbox_inches='tight', dpi=300)#

#plt.savefig(f'/mydata/deepcloud/yves/results_git/crossAttentionExp_difference.png', bbox_inches='tight', dpi=300)

# plt.savefig(f'/mydata/deepcloud/yves/results_git/crossAttention_clean_difference_50.png', bbox_inches='tight', dpi=300)
plt.savefig(f'/mydata/deepcloud/yves/results_git/HeightDepSigmoid_Difference_5S_afno_diffNorm.png', bbox_inches='tight', dpi=300)#
