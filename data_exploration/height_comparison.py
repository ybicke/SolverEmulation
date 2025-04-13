import os
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import xarray as xr

def compare_heights_at_coordinates(file_path):
    """Compare height data at specific coordinates to check for variations"""
    print('\n' + '='*50)
    print(f'COMPARING HEIGHTS AT SPECIFIC COORDINATES')
    print('='*50)
    print(f'File: {file_path}')
    
    try:
        # Open the file with netCDF4
        with Dataset(file_path, 'r') as fh:
            if 'z_ifc' not in fh.variables:
                print("Error: Variable 'z_ifc' not found in dataset!")
                return
            
            # Get the z_ifc variable
            z_ifc_var = fh.variables['z_ifc']
            shape = z_ifc_var.shape
            
            print(f"z_ifc shape: {shape}")
            
            # Get coordinates information if available
            lat_var = None
            lon_var = None
            
            for var_name in fh.variables:
                if var_name.lower() in ['lat', 'latitude', 'clat']:
                    lat_var = fh.variables[var_name]
                    print(f"Found latitude variable: {var_name}, shape: {lat_var.shape}")
                    
                if var_name.lower() in ['lon', 'longitude', 'clon']:
                    lon_var = fh.variables[var_name]
                    print(f"Found longitude variable: {var_name}, shape: {lon_var.shape}")
            
            # Define specific coordinates to examine
            # Check corners and center of the grid
            if shape[1] > 1 and shape[2] > 1:
                coordinates = [
                    (0, 0),                     # Top-left
                    (0, shape[2]-1),            # Top-right
                    (shape[1]//2, shape[2]//2), # Center
                    (shape[1]-1, 0),            # Bottom-left
                    (shape[1]-1, shape[2]-1)    # Bottom-right
                ]
            else:
                # Fallback for 1D coordinates
                coordinates = [(0, 0)]
                
            # Collect heights at each coordinate
            height_data = {}
            
            for coord in coordinates:
                lat_idx, lon_idx = coord
                
                # Extract actual lat/lon values if available
                lat_str = f"{lat_idx}"
                lon_str = f"{lon_idx}"
                
                if lat_var is not None and lon_var is not None:
                    try:
                        if len(lat_var.shape) == 1:
                            lat_val = lat_var[lat_idx]
                            lon_val = lon_var[lon_idx]
                        else:
                            lat_val = lat_var[lat_idx, lon_idx]
                            lon_val = lon_var[lat_idx, lon_idx]
                        lat_str = f"{lat_idx} (lat: {lat_val:.2f}°)"
                        lon_str = f"{lon_idx} (lon: {lon_val:.2f}°)"
                    except Exception as e:
                        print(f"Could not extract actual lat/lon values: {e}")
                
                # Extract heights at this coordinate
                heights = z_ifc_var[:, lat_idx, lon_idx]
                height_data[coord] = heights
                
                # Print some basic info
                print('\n' + '-'*40)
                print(f"Heights at lat_idx={lat_str}, lon_idx={lon_str}:")
                print(f"  Min height: {np.min(heights):.2f} m")
                print(f"  Max height: {np.max(heights):.2f} m")
                
                # Print first few and last few levels
                num_to_show = min(5, len(heights))
                print(f"\n  First {num_to_show} levels:")
                for i in range(num_to_show):
                    print(f"    Level {i}: {heights[i]:.2f} m")
                    
                print(f"\n  Last {num_to_show} levels:")
                for i in range(len(heights) - num_to_show, len(heights)):
                    print(f"    Level {i}: {heights[i]:.2f} m")
            
            # Compare heights at different coordinates
            print('\n' + '='*50)
            print('COORDINATE COMPARISON RESULTS')
            print('='*50)
            
            reference_coord = coordinates[0]
            reference_heights = height_data[reference_coord]
            
            all_identical = True
            for coord, heights in height_data.items():
                if coord == reference_coord:
                    continue
                    
                # Check if heights are identical to reference
                if np.array_equal(heights, reference_heights):
                    print(f"Heights at coordinate {coord} are IDENTICAL to reference {reference_coord}")
                else:
                    all_identical = False
                    # Calculate differences
                    diffs = heights - reference_heights
                    max_diff = np.max(np.abs(diffs))
                    max_diff_idx = np.argmax(np.abs(diffs))
                    
                    print(f"Heights at coordinate {coord} DIFFER from reference {reference_coord}")
                    print(f"  Maximum difference: {max_diff:.6f} m at level {max_diff_idx}")
                    print(f"  Reference height at level {max_diff_idx}: {reference_heights[max_diff_idx]:.6f} m")
                    print(f"  This coordinate height at level {max_diff_idx}: {heights[max_diff_idx]:.6f} m")
                    
                    # Calculate summary statistics
                    nonzero_diffs = diffs[diffs != 0]
                    if len(nonzero_diffs) > 0:
                        print(f"  Number of differing levels: {len(nonzero_diffs)} out of {len(heights)}")
                        print(f"  Mean abs difference: {np.mean(np.abs(nonzero_diffs)):.6f} m")
                        print(f"  Std deviation of differences: {np.std(np.abs(nonzero_diffs)):.6f} m")
            
            # Plot comparison of first 3 coordinates
            plt.figure(figsize=(14, 10))
            
            # Plot absolute height values
            plt.subplot(2, 1, 1)
            for i, coord in enumerate(coordinates[:min(3, len(coordinates))]):
                plt.plot(height_data[coord], marker='.', linestyle='-', 
                         label=f"Coordinate {coord}")
            plt.title('Height Values Comparison')
            plt.xlabel('Level Index')
            plt.ylabel('Height (m)')
            plt.legend()
            plt.grid(True)
            
            # Plot differences from reference
            plt.subplot(2, 1, 2)
            for i, coord in enumerate(coordinates[1:min(4, len(coordinates))]):
                diff = height_data[coord] - reference_heights
                plt.plot(diff, marker='.', linestyle='-', 
                         label=f"Coord {coord} - Ref {reference_coord}")
            plt.title('Height Differences from Reference Coordinate')
            plt.xlabel('Level Index')
            plt.ylabel('Difference (m)')
            plt.legend()
            plt.grid(True)
            
            # Add horizontal line at y=0 for reference
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('height_coordinates_comparison.png')
            print(f"\nPlot saved as 'height_coordinates_comparison.png'")
            
            # Return overall result
            return all_identical, height_data
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        
        # Try with xarray as fallback
        print("\nAttempting analysis with xarray instead...")
        
        try:
            ds = xr.open_dataset(file_path)
            if 'z_ifc' in ds.data_vars:
                z_ifc = ds['z_ifc']
                print(f"z_ifc dimensions (xarray): {z_ifc.dims}")
                print(f"z_ifc shape: {z_ifc.shape}")
                
                # Simplified comparison with just a few points
                if len(z_ifc.shape) >= 3:
                    print('\nComparing height values at a few points:')
                    
                    shape = z_ifc.shape
                    ref_point = (0, 0)
                    ref_heights = z_ifc[:, 0, 0].values
                    
                    # Print first few and last few levels at reference point
                    print(f"\nHeights at reference point (0,0):")
                    num_to_show = min(5, len(ref_heights))
                    print(f"  First {num_to_show} levels:")
                    for i in range(num_to_show):
                        print(f"    Level {i}: {ref_heights[i]:.2f} m")
                    
                    print(f"\n  Last {num_to_show} levels:")
                    for i in range(len(ref_heights) - num_to_show, len(ref_heights)):
                        print(f"    Level {i}: {ref_heights[i]:.2f} m")
                    
                    # Check at other points (center, opposite corner)
                    check_points = [
                        (shape[1]//2, shape[2]//2),
                        (shape[1]-1, shape[2]-1)
                    ]
                    
                    for point in check_points:
                        if point[0] < shape[1] and point[1] < shape[2]:
                            other_heights = z_ifc[:, point[0], point[1]].values
                            
                            print(f"\nHeights at point {point}:")
                            print(f"  First {num_to_show} levels:")
                            for i in range(num_to_show):
                                print(f"    Level {i}: {other_heights[i]:.2f} m")
                            
                            # Compare with reference
                            if np.array_equal(ref_heights, other_heights):
                                print(f"  Heights are IDENTICAL to reference point")
                            else:
                                diffs = other_heights - ref_heights
                                max_diff = np.max(np.abs(diffs))
                                print(f"  Heights DIFFER from reference point")
                                print(f"  Maximum difference: {max_diff:.6f} m")
                
                ds.close()
                
            else:
                print("Variable 'z_ifc' not found in dataset (xarray)!")
                ds.close()
                
        except Exception as xe:
            print(f"Error with xarray approach: {xe}")
            
        return False, None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Default path
        file_path = "/mydata/deepcloud/yves/SolverEmulation/data_exploration/ml_ecrad_ape_R2B05_myrunscript_ecRad5d_infero5d_70lev_atm_3d_ICONGRID_DOM01_ml_lonlat.nc"
    
    # Run the height comparison
    compare_heights_at_coordinates(file_path) 