import torch
import pickle
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter

# -----------------------------------------------------------------------------
#   Libraries & global matplotlib setup
# -----------------------------------------------------------------------------

# Use a clean scientific style
plt.style.use('seaborn-v0_8-whitegrid')  # requires Matplotlib >=3.7; fallback ok

# Increase default font sizes a bit
plt.rcParams.update({
    'axes.titlesize': 13,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

models = [
    {
        'name': 'AFNO',
        'path': '/mydata/deepcloud/yves/results-temp/afno_column_1percent_Emb128_clean_tendency_normTarg/test'
    },
    
        # { 'name': 'RF-concat-tendency-norm',        'path': '/mydata/deepcloud/shared/results-temp/afno_column_1percent_Emb128_clean_tendency_RF/test'},
    
    
]

# Containers for plotting
y_pred_hs = []   # will hold one (height, channels) tensor per model
y_true_h = None  # will store the ground‑truth tensor (height, channels) once
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
    sample_idx = [112]                 # change this line to taste
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

    # ------------------------------------------------------------------
    # Collapse the (optional) sample dimension so that we have one curve
    # per height level. If several indices are given we take the mean over
    # those selected samples; if only one index is given we simply squeeze
    # that dimension.
    # ------------------------------------------------------------------
    if y_true.ndim == 3 and y_true.shape[0] > 1:
        y_true_h_tmp = torch.mean(y_true, dim=0)  # (H, C)
        y_pred_h_tmp = torch.mean(y_pred, dim=0)
    else:
        y_true_h_tmp = y_true.squeeze(0)  # (H, C)
        y_pred_h_tmp = y_pred.squeeze(0)

    # store predictions; ground truth needs to be stored only once
    if y_true_h is None:
        y_true_h = y_true_h_tmp
    y_pred_hs.append(y_pred_h_tmp)

    # ------------------------------------------------------------------
    # MAE calculation kept for reference (might still be useful) – comment
    # out if you no longer need it.
    # y_mae_h = torch.mean(torch.abs(y_true - y_pred), dim=0)
    # y_mae_hs.append(y_mae_h)
    
   

target_units = {
    "Sum of Temperature Tendency": "K s-1", 
    "Dynamical Temperature Tendency": "K s-1",
    "Sum of Zonal Wind Tendency": "m s-2",
    "Sum of Meridional Wind Tendency": "m s-2",
    "Convective Tend. Absolute Humidity": "kg m-3 s-1",
    "Convective Tend. Cloud Water Mass Density": "kg m-3 s-1",
    "Convective Tend. Cloud Ice Mass Density": "kg m-3 s-1"
}

# ------------------------------------------------------------------
#                      P L O T    T R U T H   &   P R E D
# ------------------------------------------------------------------

# Prepare the figure
models_name = [model['name'] for model in models]
mask = [True] * len(models_name)

# Height indices (0 ... H-1)
height_range = range(y_true_h.shape[0])

fig = plt.figure(figsize=(12, 18))

# Create one subplot for each of the 7 target channels in a 3 × 3 grid
ax_list = []

for i, (label, units) in enumerate(target_units.items()):
    subplot_idx = (3, 3, i + 1)

    ax = fig.add_subplot(*subplot_idx)

    # Plot TRUE (black solid) – label only once (first panel) so legend keys
    # are not duplicated.
    true_label = 'True' if i == 0 else None
    ax.plot(
        y_true_h[:, i],
        height_range,
        label=true_label,
        color='black',
        linewidth=2,
    )

    # ---- plot PREDICTIONS for each model ----
    for y_pred_h, model_name, msk in zip(y_pred_hs, models_name, mask):
        if msk:
            pred_label = f'Pred • {model_name}' if i == 0 else None
            ax.plot(
                y_pred_h[:, i],
                height_range,
                label=pred_label,
                linestyle='--',
            )

    ax.grid(True)
    ax.invert_yaxis()
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_title(label, fontsize=12, pad=6)
    if i in [0, 3, 6]:
        ax.set_ylabel('Height index', fontsize=14)

    # ---- Setup scientific notation with proper formatting ----
    # 1. Use ScalarFormatter with mathtext for proper scientific notation
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    # 2. Force scientific notation regardless of magnitude
    formatter.set_powerlimits((-1, 1))
    ax.xaxis.set_major_formatter(formatter)
    
    # 3. Add descriptive label with properly formatted units
    ax.set_xlabel(f'[$\\mathrm{{{units}}}$]')
    
    # 4. Set appropriate number of ticks (not too crowded, not too sparse)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    
    # 5. Calculate appropriate axis limits for this specific subplot
    # Get min/max values for this channel
    data_values = [y_true_h[:, i]]
    for pred in y_pred_hs:
        data_values.append(pred[:, i])
    
    min_val = min(d.min().item() for d in data_values)
    max_val = max(d.max().item() for d in data_values)
    
    # Add 5% padding to the range
    range_val = max_val - min_val
    min_val = min_val - 0.05 * range_val
    max_val = max_val + 0.05 * range_val
    
    # Set the limits
    ax.set_xlim(min_val, max_val)

    ax_list.append(ax)

# ------------------------------ Legend & Layout ------------------------------

# Create a single legend at the bottom centered across the figure with clear labels
# for true values and predicted values

handles, labels = [], []
for ax in ax_list:
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in labels:
            handles.append(h)
            labels.append(l)

# Place legend below all subplots
fig.legend(handles, labels, loc='lower center', ncol=len(labels),
           bbox_to_anchor=(0.5, 0.0), frameon=False)

# Tight layout with extra space at bottom for legend
fig.tight_layout(rect=[0, 0.05, 1, 1])



# Save the figure
plt.savefig('/mydata/deepcloud/yves/Tendency-normTarget-true_vs_pred-sample.png',
            bbox_inches='tight', dpi=300)


