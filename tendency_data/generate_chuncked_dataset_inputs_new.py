#!/usr/bin/env python3
"""
This script processes ICON (Icosahedral Nonhydrostatic) weather model grid data:
1. Reads input and output NetCDF files per time step
2. Extracts specified features (like temperature, pressure, etc.)
3. Interpolates 'w' from half levels to full levels and adds it to input features
4. Converts the data into HDF5 format for efficient storage and access
5. Saves each sample as a separate H5 file
"""

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
    description='Load, preprocess, and save NetCDF files in HDF5 format',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# Define command line arguments
parser.add_argument(
    '-i', '--input-nc-file', nargs='+',
    help='Input NetCDF files for features (can be multiple)',
    required=True,
    type=str
)

parser.add_argument(
    '-o', '--output-nc-file', nargs='+',
    help='Output NetCDF files for temperature (can be multiple)',
    required=True,
    type=str
)

parser.add_argument(
    '-l', '--height-layers', type=int, default=71,
    help='Number of height layers (1-70)'
)
parser.add_argument(
    '-s', '--save-path', type=str, default='.', help='Path to save output files'
)

parser.add_argument(
    '-f', '--features', nargs='+',
    default=[
        # reading order default
        'u',                # Zonal wind
        'v',                # Meridional wind
        'w',                # Vertical velocity (will be used both as target and interpolated input)
        'pres',             # Pressure
        'geopot',           # Geopotential
        'qc',               # Cloud water content
        'qi',               # Cloud ice content
        'qv',               # Specific humidity
        'clc',              # Cloud cover
        'temp',             # Temperature
        'pres_sfc',         # Surface pressure
        'cosmu0',           # Cosine of solar zenith angle
        'qv_s',             # Surface specific humidity

    ]
)

args = parser.parse_args()  # Parse the arguments

def interpolate_w_to_full_levels(w):
    """Linear interpolation from half levels to full levels."""
    # w shape: (ncells, num_half_levels)
    # Output shape: (ncells, num_full_levels)
    w_full = 0.5 * (w[:, :-1] + w[:, 1:])
    return w_full

def load_ncfile(input_file, output_file, features, height_layers):
    """
    Reads and processes NetCDF files to extract features.
    Creates separate HDF5 files for each time step.
    """
    with Dataset(input_file, mode='r') as ds_in, Dataset(output_file, mode='r') as ds_out:
        # Get time values - ensure they are the same in both files
        time_values_in = ds_in['time'][:]
        time_values_out = ds_out['time'][:]

        if not np.array_equal(time_values_in, time_values_out):
            raise ValueError(f"Time steps do not match between {input_file} and {output_file}")

        time_values = time_values_in  # Use time values from input file

        # Define feature categories
        x3d_features = [
            'u', 'v', 'pres', 'geopot', 'qc', 'qi', 'qv', 'clc', 'temp'
            # Note: 'w' will be added to x3d after interpolation. It's at the third position
        ]
        """ 
        This is the order of the x3d features in the file:
        --------------------------------------------------
        "u", "v", "w_interpolated", "pres", "geopot", "qc", "qi", "qv", "clc", "temp" 
        """

        
        x2d_features = ['pres_sfc', 'cosmu0', 'qv_s']
        w_feature = ['w']  # Vertical velocity as target

        num_timesteps = ds_in[features[0]].shape[0]
        print(f"Processing {num_timesteps} time steps")

        # Process each time step separately
        for t in range(num_timesteps):
            timestep_data = {
                'x3d': [],
                'x2d': [],
                'w': []
            }

            # Process each feature
            for feat in features:
                if feat == 'temp':
                    # Read temperature from output file
                    data = np.ma.getdata(ds_out[feat][t]).astype(np.float32)
                elif feat in ds_in.variables:
                    # Read feature from input file
                    data = np.ma.getdata(ds_in[feat][t]).astype(np.float32)
                else:
                    raise KeyError(f"Feature '{feat}' not found in input or output files")

                if feat in x3d_features:
                    # data shape is (height, ncells)
                    data = data.T  # Transpose to (ncells, height)
                    timestep_data['x3d'].append(data)
                elif feat in x2d_features:
                    # data shape is (ncells,)
                    timestep_data['x2d'].append(data)
                elif feat in w_feature:
                    # data shape is (height_2, ncells)
                    data = data.T  # Transpose to (ncells, height_2)
                    # Append original 'w' to outputs
                    timestep_data['w'].append(data)
                    # Interpolate 'w' and add to 'x3d' features
                    w_interp = interpolate_w_to_full_levels(data)
                    timestep_data['x3d'].append(w_interp)
                else:
                    raise KeyError(f"Feature '{feat}' not categorized properly")

            # Stack features for this time step with correct dimensions
            if timestep_data['x3d']:
                # Stack along new axis to get (ncells, height, num_features)
                timestep_data['x3d'] = np.stack(timestep_data['x3d'], axis=2)
            if timestep_data['x2d']:
                # Stack along new axis to get (ncells, num_features)
                timestep_data['x2d'] = np.stack(timestep_data['x2d'], axis=1)
            if timestep_data['w']:
                # Stack along new axis to get (ncells, height_2, num_features)
                timestep_data['w'] = np.stack(timestep_data['w'], axis=2)

            yield t, time_values[t], timestep_data

def generate_dataset(input_file, output_file, features, save_path, height_layers=71):
    """Generate HDF5 datasets from NetCDF files, one per time step."""
    bname_in = os.path.splitext(os.path.basename(input_file))[0]
    bname_out = os.path.splitext(os.path.basename(output_file))[0]
    logger.info(f'Processing {input_file} and {output_file}...')

    # Process each time step
    for timestep, time_value, data in load_ncfile(input_file, output_file, features, height_layers):
        # Create HDF5 file for this time step with time value in name
        h5_name = os.path.join(save_path, f'{bname_in}_idx{timestep:04d}_time_{time_value:.9f}.h5')
        with h5py.File(h5_name, 'w') as f:
            # Store time value as attribute
            f.attrs['time'] = time_value
            f.attrs['time_units'] = 'hours since 2001-1-7 00:00:00'

            for key in ['x3d', 'x2d', 'w']:
                if isinstance(data[key], list):
                    continue  # Skip if data list is empty
                if len(data[key]) > 0:
                    f.create_dataset(
                        key,
                        data=data[key],
                        # compression parameters can be adjusted if needed
                        # compression='gzip',
                        # compression_opts=9
                    )

        if timestep % 80 == 0:
            logger.info(f'Processed time step {timestep}, time: {time_value}')

def main():
    logger.info('Code started.')
    os.makedirs(args.save_path, exist_ok=True)

    # Ensure that the number of input and output files match
    if len(args.input_nc_file) != len(args.output_nc_file):
        raise ValueError("The number of input and output NetCDF files must be the same.")

    # Process each pair of input and output files
    for input_file, output_file in zip(args.input_nc_file, args.output_nc_file):
        generate_dataset(
            input_file=input_file,
            output_file=output_file,
            features=args.features,
            save_path=args.save_path,
            height_layers=args.height_layers
        )
    logger.info('Code ended.')

if __name__ == '__main__':
    main()