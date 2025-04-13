import os
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import xarray as xr

def analyze_heights(file_path):
    """Analyze height data (z_ifc) and check if heights vary across coordinates"""
    # Print file info
    file_size = os.path.getsize(file_path)
    file_size_mb = file_size / (1024 * 1024)
    print('\n' + '='*50)
    print(f'FILE INFORMATION')
    print('='*50)
    print(f'File path: {file_path}')
    print(f'File size: {file_size_mb:.2f} MB')
    
    try:
        # Open the file with netCDF4
        with Dataset(file_path, 'r') as fh:
            # Check if z_ifc exists
            if 'z_ifc' not in fh.variables:
                print("Variable 'z_ifc' not found in dataset!")
                vars = list(fh.variables.keys())
                print(f"Available variables: {vars}")
                height_vars = [v for v in vars if any(h in v.lower() for h in ['height', 'z', 'altitude', 'lev'])]
                if height_vars:
                    print(f"\nPossible height-related variables: {height_vars}")
                return
            
            # Get dimensions and shape
            z_ifc_var = fh.variables['z_ifc']
            dims = z_ifc_var.dimensions
            shape = z_ifc_var.shape
            
            print('\n' + '='*50)
            print('HEIGHT VARIABLE INFORMATION')
            print('='*50)
            print(f"z_ifc dimensions: {dims}")
            print(f"z_ifc shape: {shape}")
            
            # Extract the full height data
            z_ifc_data = z_ifc_var[:]
            
            # Basic statistics about heights
            print('\n' + '='*50)
            print('BASIC HEIGHT STATISTICS')
            print('='*50)
            
            print(f"Number of vertical levels: {shape[0]}")
            print(f"Min height: {np.min(z_ifc_data):.2f} m")
            print(f"Max height: {np.max(z_ifc_data):.2f} m")
            
            # Extract heights at different coordinates for comparison
            print('\n' + '='*50)
            print('HEIGHT COMPARISON ACROSS COORDINATES')
            print('='*50)
            
            # Take heights at several different points
            num_points = min(5, shape[1], shape[2])
            sample_points = []
            
            for i in range(num_points):
                for j in range(num_points):
                    lat_idx = i * (shape[1] // (num_points+1))
                    lon_idx = j * (shape[2] // (num_points+1))
                    if lat_idx < shape[1] and lon_idx < shape[2]:
                        sample_points.append((lat_idx, lon_idx))
            
            # Extract heights at each sample point
            heights_at_points = {}
            
            for point in sample_points:
                lat_idx, lon_idx = point
                heights = z_ifc_data[:, lat_idx, lon_idx]
                heights_at_points[point] = heights
            
            # Check if heights are the same across coordinates
            first_point = sample_points[0]
            reference_heights = heights_at_points[first_point]
            
            all_same = True
            for point, heights in heights_at_points.items():
                if not np.array_equal(heights, reference_heights):
                    all_same = False
                    break
            
            if all_same:
                print("Heights are the same at all sampled coordinates.")
                print(f"Sampled {len(sample_points)} different points across the grid.")
                
                # Just use the first point for analysis
                heights = reference_heights
            else:
                print("Heights VARY across different coordinates!")
                print("Differences between coordinates:")
                
                for point, heights in heights_at_points.items():
                    diff = heights - reference_heights
                    if np.any(diff != 0):
                        print(f"  At lat_idx={point[0]}, lon_idx={point[1]}:")
                        print(f"    Max difference: {np.max(np.abs(diff)):.2f} m")
                
                # Use the first point for subsequent analysis
                heights = reference_heights
                print("\nUsing point lat_idx=0, lon_idx=0 for further analysis.")
                
            # Analyze height differences between levels
            height_diffs = np.diff(heights)
            
            print('\n' + '='*50)
            print('HEIGHT DIFFERENCES BETWEEN LEVELS')
            print('='*50)
            print(f"Min level-to-level difference: {np.min(height_diffs):.2f} m")
            print(f"Max level-to-level difference: {np.max(height_diffs):.2f} m")
            print(f"Mean level-to-level difference: {np.mean(height_diffs):.2f} m")
            
            # Show first few levels and their differences
            print("\nFirst 10 heights and differences:")
            for i in range(min(10, len(heights)-1)):
                print(f"  Level {i} height: {heights[i]:.2f} m → Level {i+1} height: {heights[i+1]:.2f} m (diff: {height_diffs[i]:.2f} m)")
            
            # Check if the heights are ordered monotonically
            is_increasing = np.all(height_diffs > 0)
            is_decreasing = np.all(height_diffs < 0)
            
            if is_increasing:
                print("\nHeights are monotonically INCREASING with level index.")
            elif is_decreasing:
                print("\nHeights are monotonically DECREASING with level index.")
            else:
                print("\nHeights are NOT monotonic with level index.")
                
                # Find where monotonicity is violated
                for i in range(len(height_diffs)):
                    if (is_increasing and height_diffs[i] <= 0) or (is_decreasing and height_diffs[i] >= 0):
                        print(f"  Monotonicity violation at level {i} → {i+1}: diff = {height_diffs[i]:.2f} m")
            
            # Visualize heights and differences
            plt.figure(figsize=(12, 10))
            
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
            
            # Return heights for possible further use
            return heights
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        
        # Try with xarray as fallback
        print("\nAttempting analysis with xarray instead...")
        
        try:
            ds = xr.open_dataset(file_path)
            if 'z_ifc' in ds.data_vars:
                # Get dimensions and basic info
                z_ifc = ds['z_ifc']
                print(f"z_ifc dimensions (xarray): {z_ifc.dims}")
                print(f"z_ifc shape: {z_ifc.shape}")
                
                # Check if heights vary across coordinates (simplified version)
                if len(z_ifc.shape) >= 3:
                    ref_heights = z_ifc[:,0,0].values
                    other_heights = z_ifc[:,1,1].values if z_ifc.shape[1] > 1 and z_ifc.shape[2] > 1 else None
                    
                    if other_heights is not None:
                        if np.array_equal(ref_heights, other_heights):
                            print("Heights appear to be the same across coordinates.")
                        else:
                            print("Heights VARY across different coordinates!")
                            diff = ref_heights - other_heights
                            print(f"Max difference between points: {np.max(np.abs(diff)):.2f} m")
                    
                    heights = ref_heights
                else:
                    heights = z_ifc[:].values
                
                # Continue with basic analysis as above
                print('\n' + '='*50)
                print('BASIC HEIGHT STATISTICS (XARRAY)')
                print('='*50)
                
                print(f"Number of vertical levels: {len(heights)}")
                print(f"Min height: {np.min(heights):.2f} m")
                print(f"Max height: {np.max(heights):.2f} m")
                
                # Height differences
                height_diffs = np.diff(heights)
                print('\n' + '='*50)
                print('HEIGHT DIFFERENCES BETWEEN LEVELS')
                print('='*50)
                print(f"Min level-to-level difference: {np.min(height_diffs):.2f} m")
                print(f"Max level-to-level difference: {np.max(height_diffs):.2f} m")
                print(f"Mean level-to-level difference: {np.mean(height_diffs):.2f} m")
                
                # Plot (same as above)
                plt.figure(figsize=(12, 10))
                
                plt.subplot(2, 1, 1)
                plt.plot(heights, marker='o', linestyle='-')
                plt.title('Vertical Heights (z_ifc) - xarray')
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
                plt.savefig('heights_analysis_xarray.png')
                print(f"\nPlot saved as 'heights_analysis_xarray.png'")
                
                ds.close()
                return heights
            else:
                print("Variable 'z_ifc' not found in dataset (xarray)!")
                vars = list(ds.data_vars.keys())
                print(f"Available variables: {vars}")
                ds.close()
                
        except Exception as xe:
            print(f"Error with xarray approach: {xe}")
            
        return None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Default path, update as needed
        file_path = "/mydata/deepcloud/yves/SolverEmulation/data_exploration/ml_ecrad_ape_R2B05_myrunscript_ecRad5d_infero5d_70lev_atm_3d_ICONGRID_DOM01_ml_lonlat.nc"
    
    # Run the height analysis
    analyze_heights(file_path) 