#!/usr/bin/env python3
"""
Memory-efficient visualization that processes chunked test results without loading everything into memory.
"""

import sys
sys.path.append('/mydata/deepcloud/yves/SolverEmulation/A_RadiativeFlux')

import torch
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from utils.load_test_results import process_chunked_results

def calculate_mae_per_chunk(y_true, y_pred, h_true, h_pred):
    """Calculate MAE for a single chunk"""
    # Calculate errors for this chunk
    dim = [0, 1] if len(y_true.shape) == 4 else 0
    y_mae_h = torch.mean(torch.abs(y_true - y_pred), dim=dim)
    h_mae_h = torch.mean(torch.abs(h_true - h_pred), dim=dim)
    
    # Return flipped results and sample count
    return {
        'y_mae': torch.flip(y_mae_h, [0]),
        'h_mae': torch.flip(h_mae_h, [0]),
        'n_samples': y_true.shape[0]
    }

def aggregate_chunk_results(chunk_results):
    """Aggregate MAE results from multiple chunks using weighted average"""
    total_samples = sum(result['n_samples'] for result in chunk_results)
    
    # Initialize accumulators
    y_mae_sum = torch.zeros_like(chunk_results[0]['y_mae'])
    h_mae_sum = torch.zeros_like(chunk_results[0]['h_mae'])
    
    # Weighted sum
    for result in chunk_results:
        weight = result['n_samples'] / total_samples
        y_mae_sum += result['y_mae'] * weight
        h_mae_sum += result['h_mae'] * weight
    
    return y_mae_sum, h_mae_sum

def visualize_models_memory_efficient(models):
    """
    Visualize model results with memory-efficient chunk processing
    """
    y_mae_hs, h_mae_hs = [], []
    
    for model in models:
        test_path = model['path']
        print(f'Processing test files... ({test_path})')
        
        # Process chunks without loading everything into memory
        chunk_results = process_chunked_results(
            test_path, 
            calculate_mae_per_chunk,
            result_types=['y_true', 'y_pred', 'h_true', 'h_pred']
        )
        
        # Aggregate results
        y_mae_h, h_mae_h = aggregate_chunk_results(chunk_results)
        
        y_mae_hs.append(y_mae_h)
        h_mae_hs.append(h_mae_h)
        
        print(f'Processed {len(chunk_results)} chunks, total samples: {sum(r["n_samples"] for r in chunk_results)}')
    
    return y_mae_hs, h_mae_hs

def add_supplot(fig, x, ys, id, models_name, xlabel=None, ylabel=None, 
                title=None, log_scale=True, mask=None):
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

# Example usage:
if __name__ == "__main__":
    # Define your models here
    models = [
        {'path': '/mydata/deepcloud/yves/model1/test/', 'name': 'Model 1'},
        {'path': '/mydata/deepcloud/yves/model2/test/', 'name': 'Model 2'},
        # Add more models as needed
    ]
    
    # Process models with memory efficiency
    y_mae_hs, h_mae_hs = visualize_models_memory_efficient(models)
    
    # Create plots
    models_name = [model['name'] for model in models]
    mask = [True] * len(models_name)
    
    fig = plt.figure(figsize=(12, 16))
    ax00 = add_supplot(fig, x=range(71), ys=[y[:, 3] for y in y_mae_hs], id=(2, 3, 1), models_name=models_name, title='Downward fluxed', ylabel='Shortwave', xlabel='MAE [W/m$^2$]', mask=mask)
    add_supplot(fig, x=range(71), ys=[y[:, 2] for y in y_mae_hs], id=(2, 3, 2), models_name=models_name, title='Upward fluxed', xlabel='MAE [W/m$^2$]', mask=mask)
    add_supplot(fig, x=range(70), ys=[h[:, 1] for h in h_mae_hs], id=(2, 3, 3), models_name=models_name, title='Heating rates', xlabel='MAE [K/day]', mask=mask)
    add_supplot(fig, x=range(71), ys=[y[:, 1] for y in y_mae_hs], id=(2, 3, 4), models_name=models_name, ylabel='Longwave', xlabel='MAE [W/m$^2$]', mask=mask)
    ax0 = add_supplot(fig, x=range(71), ys=[y[:, 0] for y in y_mae_hs], id=(2, 3, 5), models_name=models_name, xlabel='MAE [W/m$^2$]', mask=mask)
    add_supplot(fig, x=range(70), ys=[h[:, 0] for h in h_mae_hs], id=(2, 3, 6), models_name=models_name, xlabel='MAE [K/day]', mask=mask)
    ax00.legend(fontsize="10", loc='lower left')
    
    plt.savefig('/mydata/deepcloud/yves/final_results/1D_fluxes_MAE_small.png', bbox_inches='tight', dpi=300)
    print("Plot saved!") 