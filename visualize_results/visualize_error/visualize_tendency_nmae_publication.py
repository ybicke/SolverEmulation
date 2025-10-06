import numpy as np
import pickle
import os
from os.path import join
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import hashlib

# -----------------------------------------------------------------------------
#   Configuration for nMAE plotting
# -----------------------------------------------------------------------------

# Use a clean scientific style
plt.style.use('seaborn-v0_8-whitegrid')

plt.rcParams.update({
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'grid.alpha': 1,
    'grid.linewidth': 1,
    'axes.grid': True,
    'axes.axisbelow': True,
})

# Configuration
save_to_test_path = False

# === PLOT CONFIGURATION ===
PLOT_NAME = "GT_1D_2D_large_3D_nMAE"
SUBFOLDER = "nMAE_Tendency"

# === NORMALIZATION CONFIGURATION ===
USE_LOCAL_NORMALIZATION = True  # Set to True for triangle normalization, False for global

# Statistics file paths
GLOBAL_STATS_PATH = '/mydata/deepcloud/yves/h5_tendency_data_all/train_target_statistics.pickle'
TRIANGLE_STATS_PATH = '/mydata/deepcloud/yves/h5_tendency_data_all/triangle39_train_target_statistics_v2.pickle'

final_results_path = '/mydata/deepcloud/yves/results_final'
plot_output_dir = join(final_results_path, SUBFOLDER)
cache_dir = join(plot_output_dir, '.cache')
os.makedirs(plot_output_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

models = [
    {'name': 'ViT-1D', 'path': '/mydata/deepcloud/yves/results-new/vit_1d_128_l4_triangle/test'},
    {'name': 'GT-2D-large', 'path': '/mydata/deepcloud/yves/results-new/gt_gencast_1024_l4_large/test'},
    {'name': 'GT-3D', 'path': '/mydata/deepcloud/yves/results-new/gt_simplified_128_l4_k2_drop01_simplePosEmp_heightDecoder/test'},
]

# Short descriptive titles for publication
target_units = {
    "Total Temperature": "K s⁻¹", 
    "Dynamical Temperature": "K s⁻¹",
    "Zonal Wind": "m s⁻²",
    "Meridional Wind": "m s⁻²",
    "Convective Humidity": "kg m⁻³ s⁻¹",
    "Convective Cloud Water": "kg m⁻³ s⁻¹",
    "Convective Cloud Ice": "kg m⁻³ s⁻¹"
}

# Height level to kilometer mapping
height_km = {
    0: 65, 10: 39, 20: 25, 30: 15, 40: 8, 50: 4, 60: 1, 70: 0,
}

def load_training_statistics():
    """Load training statistics for normalization."""
    stats_path = TRIANGLE_STATS_PATH if USE_LOCAL_NORMALIZATION else GLOBAL_STATS_PATH
    
    try:
        with open(stats_path, 'rb') as f:
            stats_dict = pickle.load(f)
        
        print(f"Loaded {'triangle' if USE_LOCAL_NORMALIZATION else 'global'} training statistics from: {stats_path}")
        
        if USE_LOCAL_NORMALIZATION:
            # Triangle statistics
            if 'triangle_train_target_variance' in stats_dict:
                train_variance = stats_dict['triangle_train_target_variance']
                train_std = np.sqrt(train_variance)
                print(f"Triangle column count: {stats_dict.get('triangle_column_count', 'Unknown')}")
                return train_std
        else:
            # Global statistics
            if 'train_target_variance' in stats_dict:
                train_variance = stats_dict['train_target_variance']
                train_std = np.sqrt(train_variance)
                return train_std
        
        print("Error: Required variance key not found in statistics file")
        return None
        
    except FileNotFoundError:
        print(f"Warning: Training statistics file not found at {stats_path}")
        return None

def get_cache_path(model_path):
    """Generate cache file path based on model path hash."""
    norm_suffix = "_local" if USE_LOCAL_NORMALIZATION else "_global"
    path_hash = hashlib.md5(model_path.encode()).hexdigest()[:8]
    return join(cache_dir, f"tendency_nmae_data_{path_hash}{norm_suffix}.pkl")

def load_pickle_file(filepath):
    """Load a single pickle file."""
    with open(filepath, 'rb') as handle:
        return pickle.load(handle)

def load_model_data_parallel(model_path):
    """Load pickle files for a model in parallel."""
    files = {
        'y_true': join(model_path, 'y_true.pickle'),
        'y_pred': join(model_path, 'y_pred.pickle'),
    }
    
    # Load files in parallel
    data = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_key = {executor.submit(load_pickle_file, filepath): key 
                        for key, filepath in files.items()}
        
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                data[key] = future.result()
            except Exception as exc:
                print(f'Failed to load {key}: {exc}')
                return None
    
    return data

def compute_nmae_numpy(model_data, train_std):
    """Compute nMAE using numpy operations."""
    def to_numpy(tensor):
        if hasattr(tensor, 'numpy'):
            return tensor.numpy() if hasattr(tensor, 'cpu') else tensor.numpy()
        return np.array(tensor) if not isinstance(tensor, np.ndarray) else tensor

    y_true = to_numpy(model_data['y_true'])
    y_pred = to_numpy(model_data['y_pred'])
    
    # Compute MAE first
    if len(y_true.shape) == 4:  # Shape: [samples, areas, height, features]
        mae = np.mean(np.abs(y_true - y_pred), axis=(0, 1))
    else:  # Shape: [batch, height, features]
        mae = np.mean(np.abs(y_true - y_pred), axis=0)
    
    # Convert train_std to numpy if needed
    if hasattr(train_std, 'numpy'):
        train_std = train_std.numpy()
    
    # Handle different train_std shapes
    if len(train_std.shape) > 1:
        # If train_std is per-level, take mean across levels to get per-variable std
        sigma = np.mean(train_std, axis=0)
    else:
        sigma = train_std
    
    # Normalize MAE by training std to get nMAE
    nmae = mae / sigma
    
    return nmae

def load_or_compute_nmae(model, train_std):
    """Load nMAE from cache or compute and cache it."""
    cache_path = get_cache_path(model['path'])
    
    # Check if cached data exists and is newer than source files
    if os.path.exists(cache_path):
        try:
            cache_time = os.path.getmtime(cache_path)
            source_files = [
                join(model['path'], 'y_true.pickle'),
                join(model['path'], 'y_pred.pickle'),
            ]
            
            if all(os.path.exists(f) for f in source_files):
                newest_source = max(os.path.getmtime(f) for f in source_files)
                
                if cache_time > newest_source:
                    print(f"Loading cached nMAE for {model['name']}")
                    with open(cache_path, 'rb') as f:
                        return pickle.load(f)
        except:
            pass  # Cache invalid, recompute
    
    # Compute nMAE
    norm_type = "local" if USE_LOCAL_NORMALIZATION else "global"
    print(f"Computing {norm_type} nMAE for {model['name']}")
    model_data = load_model_data_parallel(model['path'])
    if model_data is None:
        return None
        
    nmae = compute_nmae_numpy(model_data, train_std)
    
    # Cache the results
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(nmae, f)
        print(f"Cached {norm_type} nMAE for {model['name']}")
    except:
        print(f"Failed to cache {norm_type} nMAE for {model['name']}")
    
    return nmae

def add_subplot_nmae(fig, height_vals, nmae_data_list, subplot_pos, models_names, 
                     channel_idx, title=None, ylabel=None, xlabel=None):
    """
    Plots nMAE lines for each model on the same subplot with publication styling.
    """
    ax = fig.add_subplot(*subplot_pos)
    
    colors = ['red', 'blue', 'green', 'purple', 'brown', 'pink', 'orange']
    line_styles = ['-', '-.', '--', '--', '-.', '--']
    
    # Plot nMAE for each model
    for i, (nmae_data, model_name) in enumerate(zip(nmae_data_list, models_names)):
        ax.plot(
            nmae_data[:, channel_idx],
            height_vals,
            label=model_name if channel_idx == 0 else None,
            color=colors[i % len(colors)],
            linestyle=line_styles[i % len(line_styles)],
            linewidth=1.5,
        )

    # Add reference lines for interpretation
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    
    # Add text annotations for interpretation (only on first subplot to avoid clutter)
    if channel_idx == 0:
        ax.text(0.5, 5, 'Strong', rotation=90, va='bottom', ha='right', 
                color='gray', fontsize=7, alpha=0.8)
        ax.text(1.0, 5, 'Acceptable', rotation=90, va='bottom', ha='right', 
                color='gray', fontsize=7, alpha=0.8)

    ax.grid(True, alpha=0.5, linewidth=0.5)
    ax.invert_yaxis()
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    # Only show y-axis ticks on leftmost plots
    if channel_idx not in [0, 4]:
        ax.set_yticklabels([])
        ax.tick_params(axis='y', which='both', length=0)
    else:
        yticks = np.arange(0, 71, 10)
        ax.set_yticks(yticks)
        yticklabels = [f'{tick} ({height_km.get(tick, tick)} km)' for tick in yticks]
        ax.set_yticklabels(yticklabels, fontsize=8)
    
    if title:
        ax.set_title(title, fontsize=10, pad=2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    
    # Set x-axis to show reasonable range for nMAE
    data_values = [nmae[:, channel_idx] for nmae in nmae_data_list]
    min_val = max(0, min(d.min() for d in data_values) - 0.1)  # Don't go below 0
    max_val = max(d.max() for d in data_values) + 0.1
    
    ax.set_xlim(min_val, max_val)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(8))
    
    return ax

def main():
    """Main execution function."""
    norm_type = "Local (Triangle)" if USE_LOCAL_NORMALIZATION else "Global"
    print(f"Starting {norm_type} normalized tendency nMAE visualization...")
    start_time = time.time()
    
    # Load training statistics
    train_std = load_training_statistics()
    if train_std is None:
        print("Failed to load training statistics!")
        return
    
    print(f"Using {norm_type.lower()} normalization")
    print(f"Training std shape: {train_std.shape}")
    
    # Load or compute nMAE data for all models
    nmae_data_list = []
    models_name = []
    
    for model in models:
        nmae = load_or_compute_nmae(model, train_std)
        if nmae is not None:
            nmae_data_list.append(nmae)
            models_name.append(model['name'])
    
    if not nmae_data_list:
        print("No data loaded successfully!")
        return
    
    load_time = time.time() - start_time
    print(f"Data loading/computation time: {load_time:.2f}s")
    
    # Create figure
    plot_start = time.time()
    height_range = np.arange(nmae_data_list[0].shape[0])
    fig_nmae = plt.figure(figsize=(8, 7.5))
    ax_list_nmae = []

    # Create subplots for each target channel
    for i, (label, units) in enumerate(target_units.items()):
        if i < 4:
            subplot_idx = (2, 4, i + 1)
        else:
            subplot_idx = (2, 4, i + 1)

        ax = add_subplot_nmae(
            fig=fig_nmae,
            height_vals=height_range,
            nmae_data_list=nmae_data_list,
            subplot_pos=subplot_idx,
            models_names=models_name,
            channel_idx=i,
            title=label,
            ylabel='Height index' if i in [0, 4] else None,
            xlabel=f'nMAE [-]'  # nMAE is dimensionless
        )
        
        ax_list_nmae.append(ax)

    # Add title to distinguish normalization type
    fig_nmae.suptitle(f'Normalized Mean Absolute Error ({norm_type} Normalization)', 
                      fontsize=12, y=0.98)

    # Simple legend
    handles, labels = [], []
    for ax in ax_list_nmae:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)

    # Place legend below all subplots
    fig_nmae.legend(handles, labels, loc='lower center', ncol=len(labels),
                    bbox_to_anchor=(0.5, -0.03), frameon=False, fontsize=9)

    # Layout
    fig_nmae.tight_layout(rect=[0, 0.02, 1, 0.95], pad=0.1, h_pad=1, w_pad=0.5)
    
    plot_time = time.time() - plot_start
    print(f"Plotting time: {plot_time:.2f}s")

    # Save
    norm_suffix = "_local" if USE_LOCAL_NORMALIZATION else "_global"
    output_path = join(plot_output_dir, f"{PLOT_NAME}{norm_suffix}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    
    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f}s")
    print(f"Saved {norm_type.lower()} nMAE visualization to {output_path}")

    plt.show()

if __name__ == "__main__":
    main() 