import torch
from netCDF4 import Dataset
import os

def extract_heights(file_path):
    """Simply extract heights from the NetCDF file and save as a tensor"""
    try:
        with Dataset(file_path, 'r') as fh:
            if 'z_ifc' not in fh.variables:
                print("Error: Variable 'z_ifc' not found in dataset!")
                return None
            
            # Extract heights (first column)
            heights = fh.variables['z_ifc'][:, 0, 0]
            print(f"Extracted {len(heights)} height levels")
            print(f"Height range: {heights.min():.2f} m to {heights.max():.2f} m")
            
            # Convert to tensor
            heights_tensor = torch.tensor(heights, dtype=torch.float32)
            
            return heights_tensor
            
    except Exception as e:
        print(f"Error extracting heights: {e}")
        return None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Default path
        file_path = "/mydata/deepcloud/yves/SolverEmulation/data_exploration/ml_ecrad_ape_R2B05_myrunscript_ecRad5d_infero5d_70lev_atm_3d_ICONGRID_DOM01_ml_lonlat.nc"
    
    # Extract heights
    heights = extract_heights(file_path)
    
    if heights is not None:
        # Save the tensor
        output_path = os.path.dirname(os.path.abspath(__file__))
        torch.save(heights, os.path.join(output_path, 'heights.pt'))
        print(f"Heights saved to {os.path.join(output_path, 'heights.pt')}") 