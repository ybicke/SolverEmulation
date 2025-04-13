import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from netCDF4 import Dataset
import torch

def investigate_heights(file_path):
    """Investigate heights in a NetCDF file using z_ifc variable"""
    # Print file info
    file_size = os.path.getsize(file_path)
    file_size_mb = file_size / (1024 * 1024)
    print('\n' + '='*50)
    print(f'FILE INFORMATION')
    print('='*50)
    print(f'File path: {file_path}')
    print(f'File size: {file_size_mb:.2f} MB')
    
    # Try netCDF4 direct approach first
    print('\n' + '='*50)
    print('HEIGHT INVESTIGATION (using netCDF4)')
    print('='*50)
    
    try:
        with Dataset(file_path, 'r') as fh:
            # Check if z_ifc exists
            if 'z_ifc' not in fh.variables:
                print("Variable 'z_ifc' not found in dataset!")
                vars = list(fh.variables.keys())
                print(f"Available variables: {vars}")
                height_vars = [v for v in vars if any(h in v.lower() for h in ['height', 'z', 'altitude', 'lev'])]
                if height_vars:
                    print(f"\nPossible height-related variables: {height_vars}")
                return None
            
            # Get dimensions of z_ifc
            z_ifc_var = fh.variables['z_ifc']
            print(f"\nz_ifc dimensions: {z_ifc_var.dimensions}")
            print(f"z_ifc shape: {z_ifc_var.shape}")
            
            # Extract heights
            heights = z_ifc_var[:,0,0]
            
            # Basic stats
            print(f"\nHeight levels: {len(heights)}")
            print(f"Min height: {np.min(heights):.2f} m")
            print(f"Max height: {np.max(heights):.2f} m")
            
            # Calculate differences between levels
            height_diffs = np.diff(heights)
            print(f"\nHeight differences between levels:")
            print(f"Min diff: {np.min(height_diffs):.2f} m")
            print(f"Max diff: {np.max(height_diffs):.2f} m")
            print(f"Mean diff: {np.mean(height_diffs):.2f} m")
            
            # Plot the heights and differences
            plt.figure(figsize=(10, 10))
            
            # Plot heights
            plt.subplot(2, 1, 1)
            plt.plot(heights, marker='o', linestyle='-')
            plt.title('Vertical Heights (z_ifc)')
            plt.xlabel('Level Index')
            plt.ylabel('Height (m)')
            plt.grid(True)
            
            # Plot differences
            plt.subplot(2, 1, 2)
            plt.plot(height_diffs, marker='o', linestyle='-', color='r')
            plt.title('Height Differences Between Levels')
            plt.xlabel('Level Index')
            plt.ylabel('Height Difference (m)')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('heights_analysis.png')
            print(f"\nPlot saved as 'heights_analysis.png'")
            
            return heights
    
    except Exception as e:
        print(f"Error with netCDF4 approach: {e}")
        print("Trying xarray approach instead...")
        
        # Fallback to xarray
        try:
            ds = xr.open_dataset(file_path)
            if 'z_ifc' in ds.data_vars:
                z_ifc = ds['z_ifc']
                print(f"\nz_ifc dimensions (xarray): {z_ifc.dims}")
                print(f"z_ifc shape: {z_ifc.shape}")
                
                # Extract heights
                if len(z_ifc.shape) >= 3:
                    heights = z_ifc[:,0,0].values
                else:
                    heights = z_ifc[:].values
                    
                # Same analysis as above
                print(f"\nHeight levels: {len(heights)}")
                print(f"Min height: {np.min(heights):.2f} m")
                print(f"Max height: {np.max(heights):.2f} m")
                
                # Calculate differences between levels
                height_diffs = np.diff(heights)
                print(f"\nHeight differences between levels:")
                print(f"Min diff: {np.min(height_diffs):.2f} m")
                print(f"Max diff: {np.max(height_diffs):.2f} m")
                print(f"Mean diff: {np.mean(height_diffs):.2f} m")
                
                # Plot (same as above)
                plt.figure(figsize=(10, 10))
                
                plt.subplot(2, 1, 1)
                plt.plot(heights, marker='o', linestyle='-')
                plt.title('Vertical Heights (z_ifc)')
                plt.xlabel('Level Index')
                plt.ylabel('Height (m)')
                plt.grid(True)
                
                plt.subplot(2, 1, 2)
                plt.plot(height_diffs, marker='o', linestyle='-', color='r')
                plt.title('Height Differences Between Levels')
                plt.xlabel('Level Index')
                plt.ylabel('Height Difference (m)')
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig('heights_analysis.png')
                print(f"\nPlot saved as 'heights_analysis.png'")
                
                ds.close()
                return heights
            else:
                print("Variable 'z_ifc' not found in dataset (xarray)!")
                vars = list(ds.data_vars.keys())
                print(f"Available variables: {vars}")
                ds.close()
                return None
                
        except Exception as e:
            print(f"Error with xarray approach: {e}")
            return None

def analyze_for_edge_features(heights):
    """Analyze heights for use as edge features in a GNN model"""
    if heights is None:
        print("No height data available for edge feature analysis")
        return
    
    print('\n' + '='*50)
    print('EDGE FEATURE ANALYSIS FOR GNN')
    print('='*50)
    
    # Convert to tensor
    heights_tensor = torch.tensor(heights, dtype=torch.float32)
    
    # Analyze for different skip connections (max_skip = 3)
    max_skip = 3
    
    # Plot height differences for each skip distance
    plt.figure(figsize=(12, 10))
    
    for skip in range(1, max_skip + 1):
        # Calculate differences for this skip distance
        diffs = heights_tensor[skip:] - heights_tensor[:-skip]
        
        # Statistics
        print(f"\nSkip distance {skip}:")
        print(f"  Min height difference: {diffs.min().item():.2f} m")
        print(f"  Max height difference: {diffs.max().item():.2f} m")
        print(f"  Mean height difference: {diffs.mean().item():.2f} m")
        
        # Sample values
        print(f"  Sample differences (first 5):")
        for i in range(min(5, len(diffs))):
            print(f"    Level {i} → Level {i+skip}: {diffs[i].item():.2f} m")
        
        # Plot differences
        plt.subplot(max_skip, 1, skip)
        plt.plot(diffs.numpy(), marker='.', linestyle='-')
        plt.title(f'Height Differences (Skip = {skip})')
        plt.xlabel('Level Index')
        plt.ylabel('Height Diff (m)')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('edge_features_analysis.png')
    print(f"\nEdge feature analysis plot saved as 'edge_features_analysis.png'")
    
    # Demonstrate how these would be used for a small example graph
    print('\n' + '='*50)
    print('SAMPLE GNN EDGE FEATURES')
    print('='*50)
    
    # Create a simple example with 5 nodes
    sample_size = min(5, len(heights))
    sample_heights = heights_tensor[:sample_size]
    
    print(f"Sample heights (first {sample_size} nodes):")
    for i, h in enumerate(sample_heights):
        print(f"  Node {i}: {h.item():.2f} m")
    
    print("\nSample edge features (height differences):")
    # Create a mini edge index for demonstration
    for i in range(sample_size):
        for j in range(sample_size):
            if i != j:  # No self-loops
                diff = sample_heights[j] - sample_heights[i]
                print(f"  Edge {i}→{j}: {diff.item():.2f} m")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "/mydata/deepcloud/salman/dataset/ml_ecrad_ape_R2B05_myrunscript_ecRad5d_infero5d_70lev_atm_3d_ICONGRID_DOM01_ml_lonlat.nc"
        
    # Investigate heights in the file
    heights = investigate_heights(file_path)
    
    # Analyze how these heights would be used as edge features
    if heights is not None:
        analyze_for_edge_features(heights) 