"""
This script processes ICON target variables and saves them in H5 format.
"""

# Standard imports for file handling and logging
import os
import logging
import argparse
import numpy as np
import h5py
from netCDF4 import Dataset

# Set up logging configuration
logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Command line argument parser setup
parser = argparse.ArgumentParser(
    description='Load, preprocess, and save target variables in H5 format',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# Define command line arguments
parser.add_argument(
    '-n', '--nc-file', nargs='+', 
    help='Input NetCDF files (can be multiple)',
    required=True,
    type=str
)

parser.add_argument(
    '-l', '--height-layers', type=int, default=70,
    help='number of height layers'
)

parser.add_argument(
    '-s', '--save-path', type=str, default='.',
    help='path to save output files'
)

# Define target variables
parser.add_argument(
    '-f', '--features', nargs='+',
    default=[
        'ddt_temp_sum',    # sum of temperature tendencies
        'temp',            # Temperature
        'ddt_temp_dyn',    # dynamical temperature tendency
        'ddt_u_sum',       # sum of zonal wind tendencies
        'ddt_v_sum',       # sum of meridional wind tendencies
        'ddt_qv_conv',     # convective tendency of absolute humidity
        'ddt_qc_conv',     # convective tendency of cloud water mass density
        'ddt_qi_conv',     # convective tendency of cloud ice mass density<<
    ]
)

args = parser.parse_args()

def load_ncfile(file, features, height_layers):
    """
    Reads and processes NetCDF files to extract target variables.
    """
    with Dataset(file, mode='r') as ds:
        # Get time values
        time_values = ds['time'][:]
        
        # Define feature categories
        y_features = features

        num_timesteps = ds[features[0]].shape[0]
        print(f"Processing {num_timesteps} time steps")

        # Process each time step separately
        for t in range(num_timesteps):
            timestep_data = {
                'y': []  # All features are 3D targets
            }

            # Process each feature
            for feat in y_features:
                data = np.ma.getdata(ds[feat][t]).astype(np.float32)
                # data shape is (height, ncells)
                timestep_data['y'].append(data.T)  # transpose to (ncells, height)

            # Stack features along last axis
            timestep_data['y'] = np.stack(timestep_data['y'], axis=2)
            # Final shape: (ncells, height, num_features)

            yield t, time_values[t], timestep_data

def generate_dataset(nc_file, features, save_path, height_layers=70):
    """Generate HDF5 datasets from NetCDF file, one per time step."""
    bname = os.path.splitext(os.path.basename(nc_file))[0]
    logger.info(f'Processing {nc_file}...')
    
    # Create all necessary directories in the path
    output_dir = os.path.dirname(save_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each time step
    for timestep, time_value, data in load_ncfile(nc_file, features, height_layers):
        # Create HDF5 file for this time step with time value in name
        h5_name = os.path.join(save_path, f'{bname}_idx{timestep:04d}_time_{time_value:.9f}.h5')
        with h5py.File(h5_name, 'w') as f:
            # Store time value as attribute
            f.attrs['time'] = time_value
            f.attrs['time_units'] = 'hours since 2001-1-7 00:00:00'
            
            # Store feature names as attribute
            f.attrs['feature_names'] = features
            
            # Create dataset
            f.create_dataset(
                'y',
                data=data['y'],
                #compression='gzip',
                #compression_opts=9
            )
        
        if timestep % 80 == 0:
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