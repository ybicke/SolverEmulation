import numpy as np
import pickle
import os
from os.path import join
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from functools import lru_cache
import hashlib

# -----------------------------------------------------------------------------
#   Optimized Libraries & global matplotlib setup
# -----------------------------------------------------------------------------

# Use a faster style for development
plt.style.use('default')  # Faster than seaborn

# Optimized settings - reduce DPI for faster rendering during development
plt.rcParams.update({
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
    'figure.dpi': 100,         # Reduced DPI for faster rendering
    'savefig.dpi': 300,        # High quality save
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'grid.alpha': 0.6,
    'grid.linewidth': 0.5,
    'axes.grid': True,
    'axes.axisbelow': True,
})

# === PLOT CONFIGURATION ===
PLOT_NAME = "flux_soft_thresholding_final"
SUBFOLDER = "MAE_Sparsity"

final_results_path = '/mydata/deepcloud/yves/results_final'
plot_output_dir = join(final_results_path, SUBFOLDER)
cache_dir = join(plot_output_dir, '.cache')
os.makedirs(plot_output_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

models = [
    #{'name': 'GNN', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results/gnn_128_l4/test'},
    #{'name': 'ViT', 'path': '/mydata/deepcloud/yves/results_git/vit_column_128_l4_h6_concat/test'},
    #{'name': 'AFNO', 'path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_clean/test'},
    # {'name': 'BiLSTM-medium', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results/rnn_medium/test'},
    
    
    # Fluxes 1D models hrlu
    #{'name': 'GNN', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results/gnn_1d_128_hrlu_0005/test'},
    #{'name': 'ViT', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results/vit_128_hrlu_0005_new/test'},
    #{'name': 'AFNO', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results/afno_1d_128_hrlu/test'},
    #{'name': 'BiLSTM-32-128', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results/rnn_32_128_hrlu_0005/test'},
    # {'name': 'BiLSTM-medium', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results/rnn_medium_hrlu/test'},
    
    #{'name': 'BiLSTM', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results/rnn_medium_hrlu_second/test'},
    #{'name': 'BiLSTM', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results/rnn_medium_second/test'},
    
    
    
    #{'name': 'AFNO_010', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results-new/afno_1d_64_full_1percent_sparse_010/test'},
    {'name': 'AFNO_021', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results-new/afno_1d_64_full_1percent_sparse_021/test'},
    #{'name': 'AFNO_051', 'path': '/mydata/deepcloud/yves/results_A_RadiativeFlux/results-new/afno_1d_64_full_1percent_sparse_051/test'},
    
    #{'name': 'AFNO_010', 'path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_easyConcat_00106/test'},
    #{'name': 'AFNO_021', 'path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_easyConcat_004/test'},
   # {'name': 'AFNO_051', 'path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_easyConcat_008/test'},
    
    #{'name': 'AFNO_0172', 'path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_easyConcat_00172/test'},
    #{'name': 'AFNO_0208', 'path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_easyConcat_00208/test'},
    #{'name': 'AFNO_04', 'path': '/mydata/deepcloud/yves/online-datasets/workspace/results/afno_column_1percent_Emb128_spars04/test'},
    #{'name': 'AFNO_05', 'path': '/mydata/deepcloud/yves/online-datasets/workspace/results/afno_column_1percent_Emb128_spars05/test'},

    {'name': 'AFNO_000', 'path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_easyConcat_000/test'},
    {'name': 'AFNO_0106', 'path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_easyConcat_00106/test'},
    #{'name': 'AFNO_0172', 'path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_easyConcat_00172/test'},
    #{'name': 'AFNO_0208', 'path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_easyConcat_00208/test'},
    #{'name': 'AFNO_03', 'path': '/mydata/deepcloud/yves/online-datasets/workspace/results/afno_column_1percent_Emb128_spars03/test'},

    
    #{'name': 'AFNO_06', 'path': '/mydata/deepcloud/yves/online-datasets/workspace/results/afno_column_1percent_Emb128_hard06/test'},
    #{'name': 'AFNO_07', 'path': '/mydata/deepcloud/yves/online-datasets/workspace/results/afno_column_1percent_Emb128_hard07/test'},
    #{'name': 'AFNO_08', 'path': '/mydata/deepcloud/yves/online-datasets/workspace/results/afno_column_1percent_Emb128_hard08/test'},
    #{'name': 'AFNO_09', 'path': '/mydata/deepcloud/yves/online-datasets/workspace/results/afno_column_1percent_Emb128_hard09/test'},
    
]


def get_cache_path(model_path):
    """Generate cache file path based on model path hash."""
    path_hash = hashlib.md5(model_path.encode()).hexdigest()[:8]
    return join(cache_dir, f"mae_data_{path_hash}.pkl")

def load_pickle_file(filepath):
    """Load a single pickle file."""
    with open(filepath, 'rb') as handle:
        return pickle.load(handle)

def load_model_data_parallel(model_path):
    """Load all 4 pickle files for a model in parallel."""
    files = {
        'y_true': join(model_path, 'y_true.pickle'),
        'y_pred': join(model_path, 'y_pred.pickle'),
        'h_true': join(model_path, 'h_true.pickle'),
        'h_pred': join(model_path, 'h_pred.pickle')
    }
    
    # Load files in parallel using ThreadPoolExecutor
    data = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
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
    # Convert to numpy if needed (avoid torch entirely)
    def to_numpy(tensor):
        if hasattr(tensor, 'numpy'):
            return tensor.numpy() if hasattr(tensor, 'cpu') else tensor.numpy()
        return np.array(tensor) if not isinstance(tensor, np.ndarray) else tensor
    
    y_true = to_numpy(model_data['y_true'])
    y_pred = to_numpy(model_data['y_pred'])
    h_true = to_numpy(model_data['h_true'])
    h_pred = to_numpy(model_data['h_pred'])
    
    # Compute MAE
    dim = (0, 1) if len(y_true.shape) == 4 else 0
    y_mae_h = np.mean(np.abs(y_true - y_pred), axis=dim)
    h_mae_h = np.mean(np.abs(h_true - h_pred), axis=dim)
    
    # Flip to match height convention
    y_mae_h = np.flip(y_mae_h, axis=0)
    h_mae_h = np.flip(h_mae_h, axis=0)
    
    return y_mae_h, h_mae_h

def load_or_compute_mae(model):
    """Load MAE from cache or compute and cache it."""
    cache_path = get_cache_path(model['path'])
    
    # Check if cached data exists and is newer than source files
    if os.path.exists(cache_path):
        try:
            # Check if cache is still valid
            cache_time = os.path.getmtime(cache_path)
            source_files = [
                join(model['path'], 'y_true.pickle'),
                join(model['path'], 'y_pred.pickle'),
                join(model['path'], 'h_true.pickle'),
                join(model['path'], 'h_pred.pickle')
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
        return None, None
        
    y_mae_h, h_mae_h = compute_mae_numpy(model_data)
    
    # Cache the results
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump((y_mae_h, h_mae_h), f)
        print(f"Cached MAE for {model['name']}")
    except:
        print(f"Failed to cache MAE for {model['name']}")
    
    return y_mae_h, h_mae_h

# Height level to kilometer mapping
height_km = {
    0: 65, 10: 39, 20: 25, 30: 15, 40: 8, 50: 4, 60: 1, 70: 0,
}

def add_subplot_flux_optimized(fig, x, ys, subplot_pos, models_name, xlabel=None, 
                              title=None, is_heating=False, is_flux=False):
    """Optimized subplot function with minimal styling for speed."""
    ax = fig.add_subplot(*subplot_pos)
    
    # Simple colors and styles
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
    line_styles = ['-', '--', '-.', ':', '-', '--']
    
    # Plot each model
    for i, (y, model_name) in enumerate(zip(ys, models_name)):
        ax.plot(y, x, label=model_name, color=colors[i % len(colors)],
                linestyle=line_styles[i % len(line_styles)], linewidth=1.2)
    
    ax.invert_yaxis()
    
    # Simplified axis handling
    subplot_idx = subplot_pos[2]
    if subplot_idx not in [1, 4]:
        ax.set_yticklabels([])
        ax.tick_params(axis='y', which='both', length=0)
    else:
        yticks = np.arange(0, 69 if is_heating else 71, 10)
        ax.set_yticks(yticks)
        yticklabels = [f'{tick} ({height_km.get(tick, tick)} km)' for tick in yticks]
        ax.set_yticklabels(yticklabels, fontsize=8)
        ax.set_ylabel('Height level', fontsize=9)
    
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10, pad=2)
    
    # Simplified scaling
    if is_flux:
        #ax.set_xscale('log')  # Set log scale for heating rates
        #ax.set_xlim(1e-3, 1e2) 
        #ax.xaxis.set_major_locator(ticker.LogLocator(numticks=6))
        #ax.xaxis.set_minor_locator(ticker.LogLocator(subs='all', numticks=10))
        #ax.grid(True, which='both', alpha=0.3, linewidth=0.5)
        ax.set_xlim(-0.5, 7)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
    elif is_heating:
        ax.set_xscale('log')  # Set log scale for heating rates
        ax.set_xlim(1e-2, 1e3)  # Adjust limits for log scale
        ax.xaxis.set_major_locator(ticker.LogLocator(numticks=6))
        ax.xaxis.set_minor_locator(ticker.LogLocator(subs='all', numticks=10))
        ax.grid(True, which='both', alpha=0.3, linewidth=0.5)
        
        #ax.set_xlim(-5, 30)
        #ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
    else:
        ax.set_xlim(-0.5, 95)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
    
    # Minimal grid
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    return ax

def main():
    """Main execution function."""
    print("Starting optimized flux MAE visualization...")
    start_time = time.time()
    
    # Load or compute MAE data for all models (potentially in parallel)
    y_mae_hs, h_mae_hs = [], []
    models_name = []
    
    # Process models in parallel for maximum speed
    with ThreadPoolExecutor(max_workers=min(4, len(models))) as executor:
        future_to_model = {executor.submit(load_or_compute_mae, model): model 
                          for model in models}
        
        results = []
        for future in as_completed(future_to_model):
            model = future_to_model[future]
            try:
                y_mae_h, h_mae_h = future.result()
                if y_mae_h is not None and h_mae_h is not None:
                    results.append((model, y_mae_h, h_mae_h))
            except Exception as exc:
                print(f'Model {model["name"]} failed: {exc}')
    
    # Sort results to maintain consistent order
    results.sort(key=lambda x: models.index(x[0]))
    
    for model, y_mae_h, h_mae_h in results:
        y_mae_hs.append(y_mae_h)
        h_mae_hs.append(h_mae_h)
        models_name.append(model['name'])
    
    load_time = time.time() - start_time
    print(f"Data loading/computation time: {load_time:.2f}s")
    
    # Create figure
    plot_start = time.time()
    fig = plt.figure(figsize=(6.5, 8))
    
    # Create subplots using optimized function
    ax_list = []
    
    ax1 = add_subplot_flux_optimized(fig, x=range(71), ys=[y[:, 3] for y in y_mae_hs], 
                                   subplot_pos=(2, 3, 1), models_name=models_name, 
                                   title='SW Downward Flux', xlabel='MAE [W m⁻²]', is_flux=True)
    ax_list.append(ax1)
    
    ax2 = add_subplot_flux_optimized(fig, x=range(71), ys=[y[:, 2] for y in y_mae_hs], 
                                   subplot_pos=(2, 3, 2), models_name=models_name, 
                                   title='SW Upward Flux', xlabel='MAE [W m⁻²]', is_flux=True)
    ax_list.append(ax2)
    
    ax3 = add_subplot_flux_optimized(fig, x=range(70), ys=[h[:, 1] for h in h_mae_hs], 
                                   subplot_pos=(2, 3, 3), models_name=models_name, 
                                   title='SW Heating Rates', xlabel='MAE [K day⁻¹]', is_heating=True)
    ax_list.append(ax3)
    
    ax4 = add_subplot_flux_optimized(fig, x=range(71), ys=[y[:, 1] for y in y_mae_hs], 
                                   subplot_pos=(2, 3, 4), models_name=models_name, 
                                   xlabel='MAE [W m⁻²]', title='LW Downward Flux', is_flux=True)
    ax_list.append(ax4)
    
    ax5 = add_subplot_flux_optimized(fig, x=range(71), ys=[y[:, 0] for y in y_mae_hs], 
                                   subplot_pos=(2, 3, 5), models_name=models_name, 
                                   xlabel='MAE [W m⁻²]', title='LW Upward Flux', is_flux=True)
    ax_list.append(ax5)
    
    ax6 = add_subplot_flux_optimized(fig, x=range(70), ys=[h[:, 0] for h in h_mae_hs], 
                                   subplot_pos=(2, 3, 6), models_name=models_name, 
                                   xlabel='MAE [K day⁻¹]', title='LW Heating Rates', is_heating=True)
    ax_list.append(ax6)
    
    # Simple legend
    handles, labels = ax_list[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               bbox_to_anchor=(0.5, -0.0), frameon=False, fontsize=9)
    
    # Layout
    fig.tight_layout(rect=[0, 0.05, 1, 0.92], pad=0.3, h_pad=2.0, w_pad=1.0)
    
    plot_time = time.time() - plot_start
    print(f"Plotting time: {plot_time:.2f}s")
    
    # Save
    output_path = join(plot_output_dir, f"{PLOT_NAME}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    
    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f}s")
    print(f"Saved optimized flux MAE visualization to {output_path}")
    
    plt.show()

if __name__ == "__main__":
    main() 