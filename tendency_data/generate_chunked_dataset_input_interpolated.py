

"""
This script processes ICON (Icosahedral Nonhydrostatic) weather model grid data:
1. Reads two .nc (NetCDF) files - one for current time and one for 3-min delta
2. Extracts specified atmospheric/weather features (like temperature, pressure, etc.)
3. Converts the data into HDF5 format for efficient storage and access
4. Saves each sample as a separate H5 file
"""

# Standard imports for file handling and logging
import os
import logging
import argparse
import pickle

# Scientific and ML libraries
import numpy as np
import h5py  # Replace tensorflow with h5py
from netCDF4 import Dataset

# Set up logging configuration
logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Command line argument parser setup
parser = argparse.ArgumentParser(
    description='Load, preprocess, and save .nc files in .npy format',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# Define command line arguments
parser.add_argument(
    '-n', '--nc-file', nargs='+', 
    help='Input NetCDF files (can be multiple)',
    required=True,
    type=str  # Explicitly specify string type
)

parser.add_argument(
    '-l', '--height-layers', type=int, default=71,
    help='number of height layers (1-70)'
)
parser.add_argument(
    '-s', '--save-path', type=str, default='.', help='path to save output files'
)

parser.add_argument(
    '-f', '--features', nargs='+',
    default=[
        # Wind components
        'u',                # Zonal wind
        'v',                # Meridional wind
        'w',                # Vertical velocity
        
        # Pressure and potential
        'geopot',          # Geopotential
        'pres_sfc',        # Surface pressure
        'pres',            # Pressure
        
        # Solar and radiation
        'cosmu0',          # Cosine of solar zenith angle
        # 'emis_rad',      # Surface emissivity
        
        # Cloud and humidity variables
        'clc',             # Cloud cover
        'qv_s',            # Surface specific humidity
        'qc',              # Cloud water content
        'qi',              # Cloud ice content
        'qv',              # Specific humidity
        
        # Boundary coordinates (if needed)
        #'clon_bnds',       # Longitude boundaries
        #'clat_bnds',       # Latitude boundaries
        #'height_bnds'      # Height boundaries
    ]
)

args = parser.parse_args()  # Parse the arguments

def load_ncfile(file, features, height_layers):
    """
    Reads and processes NetCDF files to extract features.
    Creates separate HDF5 files for each time step.
    """
    with Dataset(file, mode='r') as ds:
        # Get time values
        time_values = ds['time'][:]
        
        # Define feature categories
        x3d_features = ['u', 'v', 'w', 'pres', 'geopot', 'qc', 'qi', 'qv', 'clc']
        x2d_features = ['pres_sfc', 'cosmu0', 'qv_s']

        num_timesteps = ds[features[0]].shape[0]
        print(f"Processing {num_timesteps} time steps")

        # Process each time step separately
        for t in range(num_timesteps):
            timestep_data = {
                'x3d': [],
                'x2d': []
            }

            # Process each feature
            for feat in features:
                data = np.ma.getdata(ds[feat][t]).astype(np.float32)
                
                if feat in x3d_features:
                    if feat == 'w':
                        # Get the actual number of half height layers (71)
                        num_half_layers = data.shape[0]
                        num_full_layers = num_half_layers - 1  # 70
                        
                        # Interpolate 'w' from half height layers to full height layers
                        w_interp = np.zeros((num_full_layers, data.shape[1]), dtype=data.dtype)
                        
                        # First full level (surface): use first half-level value
                        w_interp[0] = data[1]  # First full level gets value from first half level above it
                        
                        # Interior levels: average of half levels above and below
                        w_interp[1:-1] = 0.5 * (data[1:-1] + data[2:])
                        
                        # Last full level: use last half-level value
                        w_interp[-1] = data[-2]  # Last full level gets value from last half level below it
                        
                        data = w_interp
                    
                    # data shape is (height, ncells)
                    timestep_data['x3d'].append(data.T)  # transpose to (ncells, height)
                elif feat in x2d_features:
                    # data shape is (ncells,)
                    timestep_data['x2d'].append(data)

            # Stack features for this time step with correct dimensions
            if timestep_data['x3d']:
                # Stack along new axis to get (ncells, height, num_features)
                timestep_data['x3d'] = np.stack(timestep_data['x3d'], axis=2)
            if timestep_data['x2d']:
                # Stack along new axis to get (ncells, num_features)
                timestep_data['x2d'] = np.stack(timestep_data['x2d'], axis=1)

            yield t, time_values[t], timestep_data




def generate_dataset(nc_file, features, save_path, height_layers=71):
    """Generate HDF5 datasets from NetCDF file, one per time step."""
    bname = os.path.splitext(os.path.basename(nc_file))[0]
    logger.info(f'Processing {nc_file}...')
    
    # Process each time step
    for timestep, time_value, data in load_ncfile(nc_file, features, height_layers):
        # Create HDF5 file for this time step with time value in name
        h5_name = os.path.join(save_path, f'{bname}_idx{timestep:04d}_time_{time_value:.9f}.h5')
        with h5py.File(h5_name, 'w') as f:
            # Store time value as attribute
            f.attrs['time'] = time_value
            f.attrs['time_units'] = 'hours since 2001-1-7 00:00:00'
            
            for key in ['x3d', 'x2d']:  # Removed 'y' from keys
                if len(data[key]) > 0:
                    f.create_dataset(
                        key,
                        data=data[key],
                        #compression='gzip',
                        #compression_opts=9
                    )
        
        if timestep % 10 == 0:
            logger.info(f'Processed time step {timestep}, time: {time_value}')


def main():
    logger.info('Code started.')
    os.makedirs(args.save_path, exist_ok=True)

    # Process each input file
    for nc_file in args.nc_file:
        generate_dataset(
            nc_file=nc_file,
            features=args.features,
            save_path=args.save_path,
            height_layers=args.height_layers
        )
    logger.info('Code ended.')

if __name__ == '__main__':
    main()