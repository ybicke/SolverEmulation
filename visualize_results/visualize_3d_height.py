import torch
import pickle
import matplotlib.pyplot as plt
import os
from os.path import join
# Configure models to visualize
models = [
    #{'name': 'GNN-3D-32-l3-fully', 'path': '/mydata/deepcloud/yves/A_RadiativeFlux/results/gnn3d_32_l3_fully/test'},
    #{'name': 'GNN-3D-32-l3', 'path': '/mydata/deepcloud/yves/A_RadiativeFlux/results/gnn3d_32_l3/test'},
    #{'name': 'GNN-1D-32-l3-globe', 'path': '/mydata/deepcloud/yves/A_RadiativeFlux/results/gnn_32_l3_optimized/test'},
    #{'name': 'GNN-64-l2-single-columns', 'path': '/mydata/deepcloud/yves/A_RadiativeFlux/results/gnn_64_l2/test'},

    #{'name': 'GNN-3D-32-l3-horizontal', 'path': '/mydata/deepcloud/yves/A_RadiativeFlux/results/gnn3d_1024_emb32_l3/test'},
    #{'name': 'GNN-3D-32-l3-indep', 'path': '/mydata/deepcloud/yves/A_RadiativeFlux/results/gnn3d_1024_emb32_l3_indep/test'},
    
    #{'name': 'GNN-1D-32-l3-triangle', 'path': '/mydata/deepcloud/yves/A_RadiativeFlux/results/gnn_32_l3_optimized1_1d_Triangle/test'},
    #{'name': 'GNN-3D-32-l3-triangle-indep', 'path': '/mydata/deepcloud/yves/A_RadiativeFlux/results/gnn3d_1024_emb32_l3_sigm_indep/test'},
    #{'name': 'GNN-3D-32-l3-triangle-horizontal', 'path': '/mydata/deepcloud/yves/A_RadiativeFlux/results/gnn3d_1024_emb32_l3_sigm/test'},
    
    
    
    {'name': 'GNN-64-l2-100eps','path': '/mydata/deepcloud/yves/results-temp/gnn3d_id39_tendency_64_l2_100/test'},
    #{'name': 'GNN-64-l2-100eps-indep','path': '/mydata/deepcloud/yves/results-temp/gnn3d_id39_tendency_64_l2_100/test'},
    #{'name': 'GNN-64-l2-100eps-1d-triangle','path': '/mydata/deepcloud/yves/results-temp/gnn_64_l2_tendency_1d_triangle/test'},   
    {'name': 'GNN-64-l2-100eps-noFully','path': '/mydata/deepcloud/yves/results-temp/gnn3d_id39_tendency_64_l2_100_nofully/test'},




    #{'name': 'GNN-3D-32-l3-1d-triangle', 'path': '/mydata/deepcloud/yves/A_RadiativeFlux/results/gnn_32_l3_optimized1_1d_Triangle/test'},
    
    #{'name': 'GNN-3D-32-l2', 'path': '/mydata/deepcloud/yves/A_RadiativeFlux/results/gnn3d_1024_emb32_l2/test'},
    #{'name': 'GNN-3D-64-l2', 'path': '/mydata/deepcloud/yves/A_RadiativeFlux/results/gnn3d_1024_emb64_l2/test'},
    #{'name': 'GNN-3D-128-l2', 'path': '/mydata/deepcloud/yves/A_RadiativeFlux/results/gnn3d_1024_emb128_l2/test'},
    

    
    
    
    # Add more single-column models if needed
]

def load_data(test_path):
    """Load the prediction data from pickle files"""
    print(f'Loading test files from {test_path}...')
    
    with open(join(test_path, 'y_true.pickle'), 'rb') as handle:
        y_true = pickle.load(handle)
    print(f'y_true: {y_true.shape}')
    
    with open(join(test_path, 'y_pred.pickle'), 'rb') as handle:
        y_pred = pickle.load(handle)
    print(f'y_pred: {y_pred.shape}')
    
    with open(join(test_path, 'h_true.pickle'), 'rb') as handle:
        h_true = pickle.load(handle)
    print(f'h_true: {h_true.shape}')
    
    with open(join(test_path, 'h_pred.pickle'), 'rb') as handle:
        h_pred = pickle.load(handle)
    print(f'h_pred: {h_pred.shape}')
    
    return y_true, y_pred, h_true, h_pred

def calculate_errors(y_true, y_pred, h_true, h_pred):
    """Calculate error metrics for height profiles"""
    # For 3D data with shape [batch, columns, height, channels]
    if len(y_true.shape) == 4:
        dim = [0, 1]  # Average over batch and columns
    # For 2D data with shape [batch, height, channels]
    else:
        dim = 0  # Average only over batch
        
    # Calculate MAE along height dimension
    y_mae_h = torch.mean(torch.abs(y_true - y_pred), dim=dim)
    h_mae_h = torch.mean(torch.abs(h_true - h_pred), dim=dim)
    
    # Calculate RMSE along height dimension
    y_rmse_h = torch.sqrt(torch.mean((y_true - y_pred)**2, dim=dim))
    h_rmse_h = torch.sqrt(torch.mean((h_true - h_pred)**2, dim=dim))
    
    # Flip to have surface at bottom of plots
    y_mae_h = torch.flip(y_mae_h, [0])
    h_mae_h = torch.flip(h_mae_h, [0])
    y_rmse_h = torch.flip(y_rmse_h, [0])
    h_rmse_h = torch.flip(h_rmse_h, [0])
    
    return y_mae_h, h_mae_h, y_rmse_h, h_rmse_h

def add_subplot(fig, x, ys, id, models_name, xlabel=None, ylabel=None, 
                title=None, log_scale=True, mask=None):
    """Create a subplot in the given figure"""
    ax = fig.add_subplot(*id)
    for y, model_name, msk in zip(ys, models_name, mask):
        if msk:
            ax.plot(y, x, label=f'{model_name}')
    ax.grid()
    ax.invert_yaxis()
    ax.set_ylabel(ylabel if ylabel else '')
    ax.set_xlabel(xlabel if xlabel else '')
    ax.set_title(title if title else '')
    if log_scale:
        ax.set_xscale('log')
    return ax

def main():
    # Output directory for saving plots
    output_dir = '/mydata/deepcloud/yves'
    os.makedirs(output_dir, exist_ok=True)
    
    # Lists to store error metrics for all models
    y_mae_hs, h_mae_hs = [], []
    y_rmse_hs, h_rmse_hs = [], []
    
    # Process each model
    for model in models:
        test_path = model['path']
        
        # Load data
        y_true, y_pred, h_true, h_pred = load_data(test_path)
        
        # Calculate errors
        print(f'Calculating errors for {model["name"]}...')
        y_mae_h, h_mae_h, y_rmse_h, h_rmse_h = calculate_errors(y_true, y_pred, h_true, h_pred)
        
        # Store results
        y_mae_hs.append(y_mae_h)
        h_mae_hs.append(h_mae_h)
        y_rmse_hs.append(y_rmse_h)
        h_rmse_hs.append(h_rmse_h)
    
    # Create plots
    print('Creating plots...')
    
    # MAE plot
    models_name = [model['name'] for model in models]
    mask = [True] * len(models_name)
    
    # Create MAE plot
    fig = plt.figure(figsize=(12, 16))
    ax1 = add_subplot(fig, x=range(y_mae_hs[0].shape[0]), ys=[y[:, 3] for y in y_mae_hs], 
                id=(2, 3, 1), models_name=models_name, title='Downward fluxes', 
                ylabel='Shortwave', xlabel='MAE [W/m$^2$]', mask=mask)
    add_subplot(fig, x=range(y_mae_hs[0].shape[0]), ys=[y[:, 2] for y in y_mae_hs], 
                id=(2, 3, 2), models_name=models_name, title='Upward fluxes', 
                xlabel='MAE [W/m$^2$]', mask=mask)
    add_subplot(fig, x=range(h_mae_hs[0].shape[0]), ys=[h[:, 1] for h in h_mae_hs], 
                id=(2, 3, 3), models_name=models_name, title='Heating rates', 
                xlabel='MAE [K/day]', mask=mask)
    add_subplot(fig, x=range(y_mae_hs[0].shape[0]), ys=[y[:, 1] for y in y_mae_hs], 
                id=(2, 3, 4), models_name=models_name, ylabel='Longwave', 
                xlabel='MAE [W/m$^2$]', mask=mask)
    add_subplot(fig, x=range(y_mae_hs[0].shape[0]), ys=[y[:, 0] for y in y_mae_hs], 
                id=(2, 3, 5), models_name=models_name, xlabel='MAE [W/m$^2$]', mask=mask)
    add_subplot(fig, x=range(h_mae_hs[0].shape[0]), ys=[h[:, 0] for h in h_mae_hs], 
                id=(2, 3, 6), models_name=models_name, xlabel='MAE [K/day]', mask=mask)
    
    # Add legend to the first subplot
    ax1.legend(fontsize="10", loc='lower left')
    plt.suptitle('Mean Absolute Error (MAE) by Height Level', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle
    plt.savefig(join(output_dir, '3D_mae_tendency_64_100eps_horizontal_fully_vs_nonFully.png'), bbox_inches='tight', dpi=300)
    
    # Create RMSE plot
    #fig = plt.figure(figsize=(12, 16))
    #ax1 = add_subplot(fig, x=range(y_rmse_hs[0].shape[0]), ys=[y[:, 3] for y in y_rmse_hs], 
    #            id=(2, 3, 1), models_name=models_name, title='Downward fluxes', 
    #            ylabel='Shortwave', xlabel='RMSE [W/m$^2$]', mask=mask)
    #add_subplot(fig, x=range(y_rmse_hs[0].shape[0]), ys=[y[:, 2] for y in y_rmse_hs], 
    #            id=(2, 3, 2), models_name=models_name, title='Upward fluxes', 
    #            xlabel='RMSE [W/m$^2$]', mask=mask)
    #add_subplot(fig, x=range(h_rmse_hs[0].shape[0]), ys=[h[:, 1] for h in h_rmse_hs], 
    #            id=(2, 3, 3), models_name=models_name, title='Heating rates', 
    #            xlabel='RMSE [K/day]', mask=mask)
    #add_subplot(fig, x=range(y_rmse_hs[0].shape[0]), ys=[y[:, 1] for y in y_rmse_hs], 
    #            id=(2, 3, 4), models_name=models_name, ylabel='Longwave', 
    #            xlabel='RMSE [W/m$^2$]', mask=mask)
    #add_subplot(fig, x=range(y_rmse_hs[0].shape[0]), ys=[y[:, 0] for y in y_rmse_hs], 
    #            id=(2, 3, 5), models_name=models_name, xlabel='RMSE [W/m$^2$]', mask=mask)
    #add_subplot(fig, x=range(h_rmse_hs[0].shape[0]), ys=[h[:, 0] for h in h_rmse_hs], 
    #            id=(2, 3, 6), models_name=models_name, xlabel='RMSE [K/day]', mask=mask)
    
    # Add legend to the first subplot
    #ax1.legend(fontsize="10", loc='lower left')
    #plt.suptitle('Root Mean Square Error (RMSE) by Height Level', fontsize=16)
    #plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle
    #plt.savefig(join(output_dir, '3D_height_rmse_profiles.png'), bbox_inches='tight', dpi=300)
    
    print(f'All plots saved to {output_dir}')

if __name__ == "__main__":
    main() 