

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
    '-n', '--nc-file', help='Input NetCDF file for current time'
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



parser.add_argument(
    '-l', '--height-layers', type=int, default=71,
    help='number of height layers (1-70)'
)
parser.add_argument(
    '-s', '--save-path', type=str, default='.', help='path to save output files'
)

args = parser.parse_args()

def load_ncfile(file, features, height_layers):
    """
    Reads and processes NetCDF files to extract features.
    
    Args:
        file: Path to main NetCDF file
        features: List of features to extract
        height_layers: Number of vertical layers to consider
    
    Returns:
        dataset: List of dictionaries containing features for h5
        feature_description: Dictionary describing feature shapes/types
    """
    
    with Dataset(file, mode='r') as ds:
        print(ds.variables.keys())  # Print available variables in the file
        print('-'*300)
        
        # Get number of samples from first feature's shape
        num_samples = ds[features[0]].shape[0]
        dataset = [{} for _ in range(num_samples)]
        feature_description = {}

        print(ds['time'], '+++++++++'*100)
        
        # Process each feature
        for feat in features:
            # Extract data and convert to float32
            data = np.ma.getdata(ds[feat][:]).astype(np.float32)
            
            # Add dimension if data is 2D
            if data.ndim == 2:
                data = np.expand_dims(data, axis=1)

            # Optional height layer filtering (commented out)
            # data = data[:, -height_layers:, :]

            print(feat, data.shape, '-'*100)
            num_layers = data.shape[1]
            
            # Create features for each layer and sample
            for i, l in enumerate(range(num_layers - 1, -1, -1)):
                for s in range(num_samples):
                    # Store current time data
                    dataset[s][f'{feat}_{i}'] = data[s, l, :]
                
                # Define feature descriptions for h5
                feature_description[f'{feat}_{i}'] = {
                    'shape': data[0, l, :].shape,
                    'dtype': data.dtype
                }

        return dataset, feature_description



def generate_dataset(nc_file, features, save_path, height_layers=71):
    """
    Generate HDF5 dataset from NetCDF file.
    
    Args:
        nc_file: Path to NetCDF file
        features: List of features to extract
        save_path: Path to save output files
        height_layers: Number of vertical layers to consider
    """
    bname = os.path.splitext(os.path.basename(nc_file))[0]
    logger.info(f'Processing {nc_file}...')
    
    # Load data from NetCDF file
    dataset, feature_description = load_ncfile(
        nc_file, features, height_layers
    )

    # Save feature description
    with open(os.path.join(save_path, 'feature_description.pickle'),
              'wb') as handle:
        pickle.dump(feature_description, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    # Create HDF5 files for each sample
    for s, sample in enumerate(dataset):
        h5_name = os.path.join(save_path, f'{bname}_sample{s:03}.h5')
        with h5py.File(h5_name, 'w') as f:
            # Create groups and datasets
            for feat_name, feat_data in sample.items():
                # Get feature description
                feat_desc = feature_description[feat_name]
                # Create dataset with appropriate shape and dtype
                f.create_dataset(
                    feat_name,
                    data=feat_data,
                    dtype=feat_desc['dtype'],
                    compression='gzip',
                    compression_opts=9
                )


def main():
    logger.info('Code started.')
    os.makedirs(args.save_path, exist_ok=True)

    generate_dataset(
        nc_file=args.nc_file,
        features=args.features,
        save_path=args.save_path,
        height_layers=args.height_layers
    )
    logger.info('Code ended.')


if __name__ == '__main__':
    main()