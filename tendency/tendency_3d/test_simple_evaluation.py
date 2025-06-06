#!/usr/bin/env python3

"""
Simple test script to evaluate new trained models with old-style evaluation
This will help isolate if the evaluation procedure is causing the performance difference
"""

import os
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsoluteError, MeanSquaredError

# Import your data loading and processing functions
from data_loader_3d_tendency import IconIterableDataset3DTendency
from data_utils import DataNormalizer, interpolate_w_to_full_levels

def simple_test_model(model_path, test_loader, normalizer, target_means, target_vars, device):
    """
    Simple evaluation function similar to the old training script
    """
    print(f"Loading model from: {model_path}")
    
    # Load the best model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load model architecture (you'll need to adjust this based on your model)
    from gnn_3d_tendency import GNN3dTendency
    
    # You'll need to get these parameters from the config file
    model = GNN3dTendency(
        total_cols=81920,
        grid_file_path="/mydata/deepcloud/yves/SolverEmulation/data_exploration/icon_grid_0008_R02B05_G.nc",
        triangle_id=39,
        embed_dim=64,
        depth=2,
        dropout=0.0,
        channels_in_3d=10,
        channels_in_2d=3,
        channels_out=7,
        edge_channels_in=1,
        num_height_levels=70,
        device=device,
        division_factor=4,
        fully_connected=True,
        disable_horizontal=False
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize loss and metric trackers
    test_loss = MeanSquaredError().to(device)
    test_mae = MeanAbsoluteError().to(device)

    y_true, y_pred = [], []
    
    print("Starting simple evaluation...")
    
    # SIMPLE EVALUATION LOOP (like old script)
    model.eval()
    for i, data in enumerate(test_loader):
        batch_x3, batch_x2, batch_y, batch_w = data
        batch_x3, batch_x2, batch_y, batch_w = batch_x3.to(device), batch_x2.to(device), batch_y.to(device), batch_w.to(device)
        
        # Interpolate vertical wind and concatenate to x3d
        w_full = interpolate_w_to_full_levels(batch_w)
        batch_x3_with_w = torch.cat([batch_x3, w_full], dim=-1)
        
        # Normalize test inputs
        batch_x3_norm_with_w, batch_x2_norm, _ = normalizer.normalize(batch_x3_with_w, batch_x2)
        
        # Transform targets
        def transform_targets(batch_y, means, variances, k=4, min_scale=1e-6):
            scale = torch.clamp(k * torch.sqrt(variances), min=min_scale)
            means = means.view(1, 1, 1, -1).expand_as(batch_y)
            scale = scale.view(1, 1, 1, -1).expand_as(batch_y)
            return (batch_y - means) / scale
        
        def inverse_transform_targets(y_norm, means, variances, k=4, min_scale=1e-6):
            scale = torch.clamp(k * torch.sqrt(variances), min=min_scale)
            means = means.view(1, 1, 1, -1).expand_as(y_norm)
            scale = scale.view(1, 1, 1, -1).expand_as(y_norm)
            return y_norm * scale + means
        
        batch_y_transformed = transform_targets(batch_y, target_means, target_vars)
        
        # SIMPLE INFERENCE (no timing, no GPU sync)
        with torch.no_grad():
            outputs = model(batch_x3_norm_with_w, batch_x2_norm)
            
        # Calculate loss on transformed outputs
        loss = test_loss(outputs, batch_y_transformed)
        mae = test_mae(outputs, batch_y_transformed)
        
        # Inverse transform targets for saving
        outputs_original = inverse_transform_targets(outputs, target_means, target_vars)
        
        # Collect true and predicted values
        y_true.append(batch_y.detach().cpu())
        y_pred.append(outputs_original.detach().cpu())

        if i % 100 == 99:
            print(f'batch {i+1} loss: {loss:.4f}, mae: {mae:.4f}')

    # Compute final metrics
    total_test_loss = test_loss.compute()
    total_test_mae = test_mae.compute()

    print(f'Final Results: loss: {total_test_loss:.6f}, mae: {total_test_mae:.6f}')
    
    # Concatenate predictions
    y_true = torch.cat(y_true, 0).numpy()
    y_pred = torch.cat(y_pred, 0).numpy()
    
    # Compute additional metrics
    overall_mae = np.mean(np.abs(y_true - y_pred))
    overall_mse = np.mean((y_true - y_pred) ** 2)
    
    print(f'Numpy verification: MAE: {overall_mae:.6f}, MSE: {overall_mse:.6f}')
    
    return {
        'test_loss': float(total_test_loss),
        'test_mae': float(total_test_mae), 
        'numpy_mae': overall_mae,
        'numpy_mse': overall_mse
    }

def compare_models():
    """Compare old vs new trained models using simple evaluation"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load normalization parameters
    def get_normalization_params(stats_file):
        with open(stats_file, 'rb') as f:
            stats = pickle.load(f)
            return torch.tensor(stats['mean2d'], dtype=torch.float32).to(device), \
                torch.tensor(stats['var2d'], dtype=torch.float32).to(device), \
                torch.tensor(stats['mean3d'], dtype=torch.float32).to(device), \
                torch.tensor(stats['var3d'], dtype=torch.float32).to(device)
    
    stats_file = '/mydata/deepcloud/shared/h5_tendency_all/normalizer_stats_per_feat_updated.pickle'
    mean2d, var2d, mean3d, var3d = get_normalization_params(stats_file)
    normalizer = DataNormalizer(mean2d, var2d, mean3d, var3d, device=device)
    
    # Load target statistics  
    target_stats_file = '/mydata/deepcloud/shared/h5_tendency_all/normalizer_stats_per_feat_y2_no_temp.pickle'
    with open(target_stats_file, 'rb') as f:
        target_stats = pickle.load(f)
    target_means = torch.tensor(target_stats['mean'], dtype=torch.float32).to(device)
    target_vars = torch.tensor(target_stats['var'], dtype=torch.float32).to(device)
    
    # Create test data loader (you'll need to adjust paths)
    # This is a simplified version - adjust based on your actual test data
    import glob
    import re
    
    INPUT_FILENAMES = glob.glob("/mydata/deepcloud/yves/h5_tendency_data_all/inputs/*_inputs_*.h5")
    OUTPUT_FILENAMES = glob.glob("/mydata/deepcloud/yves/h5_tendency_data_all/outputs/*_tendencies_*.h5")
    
    # Sort files
    input_time_indices = [float(re.search(r'_time_(\d+\.\d+)\.h5', f).group(1)) for f in INPUT_FILENAMES]
    output_time_indices = [float(re.search(r'_time_(\d+\.\d+)\.h5', f).group(1)) for f in OUTPUT_FILENAMES]
    
    sorted_input_time_indices = sorted(input_time_indices)
    sorted_output_time_indices = sorted(output_time_indices)
    
    sorted_input_files = [INPUT_FILENAMES[input_time_indices.index(time_index)] for time_index in sorted_input_time_indices]
    sorted_output_files = [OUTPUT_FILENAMES[output_time_indices.index(time_index)] for time_index in sorted_output_time_indices]
    
    # Test files
    test_input_files = sorted_input_files[2220:2240]  # Just a small subset for quick testing
    test_output_files = sorted_output_files[2220:2240]
    
    # Create test loader
    icon_data = IconIterableDataset3DTendency(
        triangle_id=39,
        division_factor=4,
        input_filenames=test_input_files,
        output_filenames=test_output_files,
        shuffle=False,
        cache_dir='/tmp',
        dtype='float32',
        total_cols=81920,      
    )
    
    test_loader = DataLoader(icon_data, batch_size=2, pin_memory=True, num_workers=4)
    
    # Test different model checkpoints
    models_to_test = {
        "Old Training": "/path/to/old/trained/model/best_model.pth",
        "New Training": "/path/to/new/trained/model/best_model.pth"
    }
    
    results = {}
    
    for name, model_path in models_to_test.items():
        if os.path.exists(model_path):
            print(f"\n{'='*50}")
            print(f"Testing: {name}")
            print(f"{'='*50}")
            
            try:
                results[name] = simple_test_model(model_path, test_loader, normalizer, target_means, target_vars, device)
            except Exception as e:
                print(f"Error testing {name}: {e}")
                results[name] = None
        else:
            print(f"Model not found: {model_path}")
    
    # Compare results
    print(f"\n{'='*50}")
    print("COMPARISON RESULTS")
    print(f"{'='*50}")
    
    for name, result in results.items():
        if result:
            print(f"{name}:")
            print(f"  Test Loss: {result['test_loss']:.6f}")
            print(f"  Test MAE:  {result['test_mae']:.6f}")
            print(f"  Numpy MAE: {result['numpy_mae']:.6f}")
            print()

if __name__ == "__main__":
    compare_models() 