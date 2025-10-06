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
#   Optimized Libraries & global matplotlib setup
# -----------------------------------------------------------------------------

# Use a clean scientific style (same as original for visual consistency)
plt.style.use('seaborn-v0_8-whitegrid')  # requires Matplotlib >=3.7; fallback ok

# Optimized settings - keep original styling but reduce DPI for speed
plt.rcParams.update({
    'axes.titlesize': 10,      # Reduced for publication
    'axes.labelsize': 9,       # Reduced for publication
    'xtick.labelsize': 8,      # Reduced for publication
    'ytick.labelsize': 8,      # Reduced for publication
    'legend.fontsize': 9,      # Reduced for publication
    'figure.dpi': 100,         # Reduced from 150 for faster rendering (only change)
    'savefig.dpi': 300,        # High quality save
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'grid.alpha': 1,           # Keep original grid styling
    'grid.linewidth': 1,       # Keep original grid styling
    'axes.grid': True,         # Enable grid by default
    'axes.axisbelow': True,    # Grid behind plot elements
})

# Configuration
save_to_test_path = False

# === PLOT CONFIGURATION ===
PLOT_NAME = "GT_1D_2D_3D_Thesis"
SUBFOLDER = "MAE_Tendency"

final_results_path = '/mydata/deepcloud/yves/results_final'
plot_output_dir = join(final_results_path, SUBFOLDER)
cache_dir = join(plot_output_dir, '.cache')
os.makedirs(plot_output_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

models = [

    
    
    #{'name': 'GNN-1D-64-L2-100', 'path': '/mydata/deepcloud/yves/results-temp/gnn_64_l2_tendency_1d_triangle_fully_connected/test'},
    #{'name': 'ViT-1D-64-L2-100', 'path': '/mydata/deepcloud/yves/results-temp/vit_128_l4_tendency_1d_triangle/test'},
    
    # Here with correct variance values during rescaling but dropout 0.3
    #{'name': 'GNN-1D-64-L2-100', 'path': '/mydata/deepcloud/yves/results-new/gnn_1d_tendency_64_l2_100/test'},
    #{'name': 'GNN-3D-64-L2-100', 'path': '/mydata/deepcloud/yves/results-new/gnn_3d_tendency_64_l2_100/test'},
    
    #Here without dropouts but k1 scaling, variance should be correct 
    # {'name': 'GNN-1D-64-L2-K1-100', 'path': '/mydata/deepcloud/yves/results-new/gnn_1d_tendency_64_l2_100_k1/test'},
    # {'name': 'GNN-3D-64-L2-100', 'path': '/mydata/deepcloud/yves/results-new/gnn_3d_tendency_64_l2_100_k1/test'},
    
    # here with correct variance values during rescaling everything clean
    #{'name': 'GNN-1D-64-L2-100', 'path': '/mydata/deepcloud/yves/results-new/gnn_1d_tendency_64_l2_100_k4_correct_variance/test'},
    #{'name': 'GNN-3D-64-L2','path': '/mydata/deepcloud/yves/results-new/gnn_3d_tendency_64_l2_100_k4_correct_variance/test'},
    #{'name': 'GNN-3D-64-L2-100-shuffled', 'path': '/mydata/deepcloud/yves/results-new/gnn_1d_tendency_64_l2_100_k4_correct_variance_shuffled/test'},
    #{'name': 'GNN-3D-64-L2-lonlat', 'path': '/mydata/deepcloud/yves/results-new/gnn_3d_tendency_64_l2_100_lonlat/test'},
    
    #{'name': 'GNN-3D-64-L2-lonlat-second', 'path': '/mydata/deepcloud/yves/results-new/gnn_3d_tendency_64_l2_100_lonlat_second/test'},

    
    #{'name': 'GNN-3D-128-L2-lonlat', 'path': '/mydata/deepcloud/yves/results-new/gnn_3d_tendency_128_l2_100_lonlat/test'},
    
    
    
    
    # GNN height decoder
    #{'name': 'GNN-3D-128-L2-lonlat-heightDecoder', 'path': '/mydata/deepcloud/yves/results-new/gnn_3d_tendency_128_l2_100_lonlat_heightDecoder/test'},
    #{'name': 'GNN-3D-64-L2-lonlat-heightDecoder', 'path': '/mydata/deepcloud/yves/results-new/gnn_3d_tendency_64_l2_100_lonlat_heightDecoder/test'},
    # {'name': 'GNN-3D-64-L2-lonlat-heightDecoder', 'path': '/mydata/deepcloud/yves/results-new/gnn_3d_tendency_64_l2_100_lonlat_heightDecoder/test'},
   
   
    # GNN's for Thesis
    #{'name': 'GNN-1D', 'path': '/mydata/deepcloud/yves/results-new/gnn_1d_tendency_128_l2_100/test'},
    #{'name': 'GNN-2D', 'path': '/mydata/deepcloud/yves/results-new/gnn_2d_graphcast_style_1024_l2_100/test'},
    #{'name': 'GNN-3D', 'path': '/mydata/deepcloud/yves/results-new/gnn_3d_tendency_128_l2_100_lonlat/test'},
    
    
    #{'name': 'GNN-3D-128-L2-lonlat-disable-horizontal', 'path': '/mydata/deepcloud/yves/results-new/gnn_3d_tendency_128_l2_100_lonlat_disable_horizontal/test'},

    

    
    # Here with 4sigma std
    # {'name': 'GNN-3D-4-sigma', 'path': '/mydata/deepcloud/yves/results-new/gnn_3d_tendency_64_l2_100_4sig/test'},

    # compare 1d new and old
    #{'name': 'GNN-1D-64-L2-100-new', 'path': '/mydata/deepcloud/yves/results-new/gnn_1d_tendency_64_l2_100_k4_reproduce_clamp_large_new/test'},
    #{'name': 'GNN-1D-64-L2-100-old', 'path': '/mydata/deepcloud/yves/results-new/gnn_64_l2_tendency_1d_triangle_reproduce_clamp_old/test'},
    #{'name': 'GNN-1D-64-L2-100-older', 'path': '/mydata/deepcloud/yves/results-temp/gnn_64_l2_tendency_1d_triangle_old/test'},
   # {'name': 'GNN-1D-64-L2-100-very-old', 'path': '/mydata/deepcloud/yves/results-temp/gnn_64_l2_tendency_1d_triangle_fully_connected/test'},


    #{'name': 'GNN-1D-64-L2-100-old', 'path': '/mydata/deepcloud/yves/results-temp/gnn_64_l2_tendency_1d_triangle_fully_connected/test'},
    #{'name': 'GNN-1D-64-L2-100-new', 'path': '/mydata/deepcloud/yves/results-new/gnn_1d_tendency_64_l2_100/test'},
    
    # the first is an MR4
    # {'name': 'GT-3D-simp-64-L4-k3', 'path': '/mydata/deepcloud/yves/results-new/gt_simplified_64_l4_triangle39_k3/test'},
    #{'name': 'GT-3D-enha-64-L4-k2-MR4-100', 'path': '/mydata/deepcloud/yves/results-new/gt_enhanced_64_l4_drop03_triangle39_k2/test'},
    
    #{'name': 'GT-3D-simp-64-L4-l2-MR2-100', 'path': '/mydata/deepcloud/yves/results-new/gt_simplified_64_l4_drop03_triangle39_k2/test'},
    #{'name': 'GT-3D-simp-64-L4-k2-drop01-simplePosEmp', 'path': '/mydata/deepcloud/yves/results-new/gt_simplified_64_l4_k2_drop01_simplePosEmp/test'},
    
    
    
    #ViT
    #{'name': 'ViT-1D-64-l4', 'path': '/mydata/deepcloud/yves/results-new/vit_1d_64_l4_triangle/test'},
    #{'name': 'ViT-1D', 'path': '/mydata/deepcloud/yves/results-new/vit_1d_128_l4_triangle/test'},
    
    # large GNN 3D versus GT 2D
    #{'name': 'GT-2D', 'path': '/mydata/deepcloud/yves/results-new/gt_gencast_1024_l2_triangle39_k2_new/test'},
    #{'name': 'GT-2D-large', 'path': '/mydata/deepcloud/yves/results-new/gt_gencast_1024_l4_large/test'},

    
    # {'name': 'GT-3D-large-256-L4-k3-100', 'path': '/mydata/deepcloud/yves/results-new/gt_simplified_large/test'},
    #{'name': 'GT-3D-128-L4', 'path': '/mydata/deepcloud/yves/results-new/gt_simplified_128_l4/test'},

    #{'name': 'AFNO-1D-128-L4', 'path': '/mydata/deepcloud/yves/results-new/afno_1d_128_l4_triangle/test'},
    

    #{'name': 'GT-3D-64-L4-k2-drop01-simplePosEmp', 'path': '/mydata/deepcloud/yves/results-new/gt_simplified_64_l4_k2_drop01_simplePosEmp/test'},
    #{'name': 'GT-3D-64-L4-k2-drop01-simplePosEmp-heightDecoder', 'path': '/mydata/deepcloud/yves/results-new/gt_simplified_64_l4_k2_drop01_simplePosEmp_heightDecoder/test'},
    #{'name': 'GT-3D-64-L4-heightDecoder', 'path': '/mydata/deepcloud/yves/results-new/gt_simplified_64_l4_height_decoder/test'},


    #{'name': 'GT-3D-128-L4-k2-drop01-simplePosEmp', 'path': '/mydata/deepcloud/yves/results-new/gt_simplified_128_l4_k2_drop01_simplePosEmp/test'},
    #{'name': 'GT-3D', 'path': '/mydata/deepcloud/yves/results-new/gt_simplified_128_l4_k2_drop01_simplePosEmp_heightDecoder/test'},



# the transformer models used in the thesis: 
    {'name': 'ViT-1D', 'path': '/mydata/deepcloud/yves/results-new/vit_1d_128_l4_triangle/test'},
    {'name': 'GT-2D', 'path': '/mydata/deepcloud/yves/results-new/gt_gencast_1024_l4_large/test'},
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

def get_cache_path(model_path):
    """Generate cache file path based on model path hash."""
    path_hash = hashlib.md5(model_path.encode()).hexdigest()[:8]
    return join(cache_dir, f"tendency_mae_data_{path_hash}.pkl")

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

def compute_mae_numpy(model_data):
    """Compute MAE using numpy operations only (faster than torch)."""
    def to_numpy(tensor):
        if hasattr(tensor, 'numpy'):
            return tensor.numpy() if hasattr(tensor, 'cpu') else tensor.numpy()
        return np.array(tensor) if not isinstance(tensor, np.ndarray) else tensor
    
    y_true = to_numpy(model_data['y_true'])
    y_pred = to_numpy(model_data['y_pred'])
    
    # Handle different data formats
    if len(y_true.shape) == 4:  # Shape: [samples, areas, height, features]
        y_mae_h = np.mean(np.abs(y_true - y_pred), axis=(0, 1))
    else:  # Shape: [batch, height, features]
        y_mae_h = np.mean(np.abs(y_true - y_pred), axis=0)
    
    return y_mae_h

def load_or_compute_mae(model):
    """Load MAE from cache or compute and cache it."""
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
                    print(f"Loading cached MAE for {model['name']}")
                    with open(cache_path, 'rb') as f:
                        return pickle.load(f)
        except:
            pass  # Cache invalid, recompute
    
    # Compute MAE
    print(f"Computing MAE for {model['name']}")
    model_data = load_model_data_parallel(model['path'])
    if model_data is None:
        return None
        
    y_mae_h = compute_mae_numpy(model_data)
    
    # Cache the results
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(y_mae_h, f)
        print(f"Cached MAE for {model['name']}")
    except:
        print(f"Failed to cache MAE for {model['name']}")
    
    return y_mae_h

def add_subplot_mae(fig, height_vals, mae_data_list, subplot_pos, models_names, 
                    channel_idx, title=None, ylabel=None, xlabel=None):
    """
    Plots MAE lines for each model on the same subplot with publication styling.
    (Optimized version with same visual output as original)
    """
    ax = fig.add_subplot(*subplot_pos)
    
    colors = ['red', 'blue', 'green', 'purple', 'brown', 'pink', 'orange']
    
    # Beautiful, high-contrast, colorblind-friendly palette
    #colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83', '#4B7F52', '#8B5A3C']
    #          Ocean     Berry     Amber     Crimson   Indigo    Forest    Chestnut

    
    # Define line styles for different models (same as original)
    line_styles = ['-', '-.', '--', '--', '-.', '--']  # Different styles for each model
    
    # Plot MAE for each model
    for i, (mae_data, model_name) in enumerate(zip(mae_data_list, models_names)):
        ax.plot(
            mae_data[:, channel_idx],
            height_vals,
            label=model_name if channel_idx == 0 else None,
            color=colors[i % len(colors)],
            linestyle=line_styles[i % len(line_styles)],  # Cycle through line styles
            linewidth=1.5,  # Thinner lines (same as original)
        )

    # Lighter grid (same as original)
    ax.grid(True, alpha=0.5, linewidth=0.5)  # Reduced alpha and linewidth for lighter grid
    ax.invert_yaxis()
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    # Only show y-axis ticks on leftmost plots (indices 0 and 4) (same as original)
    if channel_idx not in [0, 4]:
        ax.set_yticklabels([])  # Remove y-axis tick labels
        ax.tick_params(axis='y', which='both', length=0)  # Remove tick marks
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
    
    # Simplified scientific notation
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.xaxis.set_major_formatter(formatter)
    
    # Adjust the offset text position
    ax.xaxis.get_offset_text().set_fontsize(7)  # Smaller font for offset
    ax.xaxis.labelpad = 10  # Extra padding for x-axis label
    
    # Set appropriate number of ticks for narrow plots
    ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(8))
    
    # Fast axis limits calculation
    data_values = [mae[:, channel_idx] for mae in mae_data_list]
    min_val = min(d.min() for d in data_values)
    max_val = max(d.max() for d in data_values)
    
    # Add 5% padding
    range_val = max_val - min_val
    min_val = min_val - 0.05 * range_val
    max_val = max_val + 0.05 * range_val
    
    # Set the limits
    ax.set_xlim(min_val, max_val)
    
    return ax

def main():
    """Main execution function."""
    print("Starting optimized tendency MAE visualization...")
    start_time = time.time()
    
    # Load or compute MAE data for all models in parallel
    y_mae_hs = []
    models_name = []
    
    # Process models in parallel for maximum speed
    with ThreadPoolExecutor(max_workers=min(3, len(models))) as executor:
        future_to_model = {executor.submit(load_or_compute_mae, model): model 
                          for model in models}
        
        results = []
        for future in as_completed(future_to_model):
            model = future_to_model[future]
            try:
                y_mae_h = future.result()
                if y_mae_h is not None:
                    results.append((model, y_mae_h))
            except Exception as exc:
                print(f'Model {model["name"]} failed: {exc}')
    
    # Sort results to maintain consistent order
    results.sort(key=lambda x: models.index(x[0]))
    
    for model, y_mae_h in results:
        y_mae_hs.append(y_mae_h)
        models_name.append(model['name'])
    
    if not y_mae_hs:
        print("No data loaded successfully!")
        return
    
    load_time = time.time() - start_time
    print(f"Data loading/computation time: {load_time:.2f}s")
    
    # Create figure
    plot_start = time.time()
    height_range = np.arange(y_mae_hs[0].shape[0])
    fig_mae = plt.figure(figsize=(8, 7.5))
    ax_list_mae = []

    # Create subplots for each target channel
    for i, (label, units) in enumerate(target_units.items()):
        if i < 4:
            subplot_idx = (2, 4, i + 1)
        else:
            subplot_idx = (2, 4, i + 1)

        ax = add_subplot_mae(
            fig=fig_mae,
            height_vals=height_range,
            mae_data_list=y_mae_hs,
            subplot_pos=subplot_idx,
            models_names=models_name,
            channel_idx=i,
            title=label,
            ylabel='Height index' if i in [0, 4] else None,
            xlabel=f'MAE [{units}]'
        )
        
        ax_list_mae.append(ax)

    # Simple legend
    handles, labels = [], []
    for ax in ax_list_mae:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)

    # Place legend below all subplots
    fig_mae.legend(handles, labels, loc='lower center', ncol=len(labels),
                   bbox_to_anchor=(0.5, -0.03), frameon=False, fontsize=9)

    # Layout
    fig_mae.tight_layout(rect=[0, 0.02, 1, 0.98], pad=0.1, h_pad=1, w_pad=0.5)
    
    plot_time = time.time() - plot_start
    print(f"Plotting time: {plot_time:.2f}s")

    # Save
    output_path = join(plot_output_dir, f"{PLOT_NAME}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    
    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f}s")
    print(f"Saved optimized tendency MAE visualization to {output_path}")

    plt.show()

if __name__ == "__main__":
    main() 