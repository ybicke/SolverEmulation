import os
import xarray as xr

def print_netcdf_info(file_path):
    # Open the dataset
    ds = xr.open_dataset(file_path)
    
    # Print file size
    file_size = os.path.getsize(file_path)
    file_size_gb = file_size / (1024 ** 3)
    print('\n' + '='*50)
    print(f'FILE INFORMATION')
    print('='*50)
    print(f'File path: {file_path}')
    print(f'File size: {file_size_gb:.2f} GB')
    
    # Print global attributes
    print('\n' + '='*50)
    print('GLOBAL ATTRIBUTES')
    print('='*50)
    for attr in ds.attrs:
        print(f"{attr:25}: {ds.attrs[attr]}")
    
    # Print dimensions
    print('\n' + '='*50)
    print('DIMENSIONS')
    print('='*50)
    for dim, size in ds.dims.items():
        print(f"{dim:15}: {size:,} elements")
    
    # Print variables
    print('\n' + '='*50)
    print('VARIABLES')
    print('='*50)
    for var in ds.data_vars:
        print('\n' + '-'*40)
        print(f"Variable: {var}")
        print(f"Shape: {ds[var].shape}")
        print(f"Dimensions: {ds[var].dims}")
        print(f"Size: {ds[var].size:,} elements")
        
        # Print variable attributes
        if ds[var].attrs:
            print("Attributes:")
            for attr_name, attr_value in ds[var].attrs.items():
                print(f"  {attr_name:15}: {attr_value}")
    
    return ds

def analyze_heights(file_path):
    """Analyze height data in the NetCDF file, focusing on z_ifc variable"""
    ds = print_netcdf_info(file_path)
    
    print('\n' + '='*50)
    print('HEIGHT ANALYSIS (z_ifc)')
    print('='*50)
    
    for coord_name in single_point.coords:
        coord_data = single_point[coord_name]
        long_name = coord_data.attrs.get("long_name", "-")
        units = coord_data.attrs.get("units", "-")
        
        print(f"\n{coord_name}:")
        print(f"  Long Name: {long_name}")
        print(f"  Value: {coord_data.values}")
        print(f"  Units: {units}")
    
    ds.close()

# Usage
file_tendencies = '../../../../../mydata/deepcloud/salman/dataset/tmp/ml_ecrad_ape_R2B05_myrunscript_1year_183min_atm_3d_DOM01_ml_0001_lonlat.nc'

file_tendencies = '../../../../../s3/deepcloud/deepcloud/icon_tendencies/year1/ml_ecrad_ape_R2B05_myrunscript_1year_183min_tendencies_DOM01_ml_0001_lonlat.nc'
file_inputs = '../../../../../s3/deepcloud/deepcloud/icon_tendencies/year1/ml_ecrad_ape_R2B05_myrunscript_1year_183min_tendencies_inputs_DOM01_ml_0001_lonlat.nc'
/mydata/deepcloud/yves/SolverEmulation/data_exploration/ml_ecrad_ape_R2B05_myrunscript_ecRad5d_infero5d_70lev_atm_3d_ICONGRID_DOM01_ml_lonlat (1).nc
print_netcdf_info(file_inputs)