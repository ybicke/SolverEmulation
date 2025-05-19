"""
This script processes ICON (Icosahedral Nonhydrostatic) weather model grid data:
1. Reads two .nc (NetCDF) files - one for current time and one for 3-min delta
2. Extracts specified atmospheric/weather features (like temperature, pressure, etc.)
3. Converts the data into TFRecord format for efficient TensorFlow training
4. Saves each sample as a separate TFRecord file

Key features processed include:
- Radiation parameters (cosmu0, albedo)
- Atmospheric conditions (temperature, pressure)
- Cloud parameters (cloud cover, water content)
- Energy fluxes (longwave/shortwave radiation)
"""

# Standard imports for file handling and logging
import os
import pickle
import logging
import argparse

# Scientific and ML libraries
import numpy as np
import tensorflow as tf
from netCDF4 import Dataset  # Library for reading NetCDF files

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
    '-d', '--nc-file-d', help='Input NetCDF file for 3-minute delta/difference'
)
parser.add_argument(
    '-f', '--features', nargs='+',
    default=[
        # Radiation input parameters
        'cosmu0_ecrad_in', 'tsfctrad_ecrad_in',
        # Albedo parameters (visible and near-infrared, direct and diffuse)
        'albvisdif_ecrad_in', 'albvisdir_ecrad_in',
        'albnirdir_ecrad_in', 'albnirdif_ecrad_in',
        # Atmospheric state variables
        'qv_s_ecrad_in', 'pres_sfc_ecrad_in', 'temp_ecrad_in', 'pres_ecrad_in',
        # Cloud parameters
        'clc_ecrad_in', 'qc_ecrad_in', 'qi_ecrad_in', 'qv_ecrad_in',
        # Radiation flux outputs
        'lwflx_up_ecrad_out', 'lwflx_dn_ecrad_out',
        'swflx_up_ecrad_out', 'swflx_dn_ecrad_out'
    ]
)
parser.add_argument(
    '-l', '--height-layers', type=int, default=71,
    help='number of vertical height layers (1-70)'
)
parser.add_argument(
    '-s', '--save-path', type=str, default='.', help='path to save output files'
)

args = parser.parse_args()

def load_ncfile(file, file_d, features, height_layers):
    """
    Reads and processes NetCDF files to extract features.
    
    Args:
        file: Path to main NetCDF file
        file_d: Path to 3-min delta NetCDF file
        features: List of features to extract
        height_layers: Number of vertical layers to consider
    
    Returns:
        dataset: List of dictionaries containing TF Features
        feature_description: Dictionary describing feature shapes/types
    """
    
    with Dataset(file, mode='r') as ds1, Dataset(file_d, mode='r') as ds2:
        print(ds1.variables.keys())  # Print available variables in the file
        print('-'*300)
        
        # Get number of samples from first feature's shape
        num_samples = ds1[features[0]].shape[0]
        dataset = [{} for _ in range(num_samples)]
        feature_description = {}

        print(ds1['time'], '+++++++++'*100)
        
        # Process each feature
        for feat in features:
            # Extract data from both files and convert to float32
            data1 = np.ma.getdata(ds1[feat][:]).astype(np.float32)
            data2 = np.ma.getdata(ds2[feat][:]).astype(np.float32)
            
            # Add dimension if data is 2D
            if data1.ndim == 2 and data2.ndim == 2:
                data1 = np.expand_dims(data1, axis=1)
                data2 = np.expand_dims(data2, axis=1)

            # Optional height layer filtering (commented out)
            # data1 = data1[:, -height_layers:, :]
            # data2 = data2[:, -height_layers:, :]

            print(feat, data1.shape, '-'*100)
            num_layers = data1.shape[1]
            
            # Create features for each layer and sample
            for i, l in enumerate(range(num_layers - 1, -1, -1)):
                for s in range(num_samples):
                    # Store current time data
                    dataset[s][f'{feat}_{i}'] = tf.train.Feature(
                        float_list=tf.train.FloatList(value=data1[s, l, :])
                    )
                    # Store 3-min delta data
                    dataset[s][f'{feat}_{i}_d'] = tf.train.Feature(
                        float_list=tf.train.FloatList(value=data2[s, l, :])
                    )
                
                # Define feature descriptions for TF
                feature_description[f'{feat}_{i}'] = tf.io.FixedLenSequenceFeature(
                    [], dtype=tf.float32, allow_missing=True
                )
                feature_description[f'{feat}_{i}_d'] = tf.io.FixedLenSequenceFeature(
                    [], dtype=tf.float32, allow_missing=True
                )

        return dataset, feature_description

def generate_dataset(nc_file, nc_file_d, features, save_path, height_layers=71):
    """
    Main function to generate and save TFRecord dataset
    
    1. Loads and processes NetCDF files
    2. Saves feature descriptions to pickle file
    3. Creates individual TFRecord files for each sample
    """
    
    # Get base filename without extension
    bname = os.path.splitext(os.path.basename(nc_file))[0]
    logger.info(f'Processing {nc_file}...')
    
    # Load and process the data
    dataset, feature_description = load_ncfile(
        nc_file, nc_file_d, features, height_layers
    )

    # Save feature description for later use
    with open(os.path.join(save_path, 'feature_description.pickle'), 'wb') as handle:
        pickle.dump(feature_description, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Create individual TFRecord files for each sample
    for s, sample in enumerate(dataset):
        tfrecord_name = os.path.join(save_path, f'{bname}_sample{s:03}.tfrecord')
        with tf.io.TFRecordWriter(tfrecord_name) as file_writer:
            # Convert to TFRecord format and write
            record_bytes = tf.train.Example(
                features=tf.train.Features(feature=sample)
            ).SerializeToString()
            file_writer.write(record_bytes)

def main():
    """
    Main execution function:
    1. Creates output directory
    2. Processes the dataset
    """
    logger.info('Code started.')
    os.makedirs(args.save_path, exist_ok=True)

    generate_dataset(
        nc_file=args.nc_file,
        nc_file_d=args.nc_file_d,
        features=args.features,
        save_path=args.save_path,
        height_layers=args.height_layers
    )
    logger.info('Code ended.')

if __name__ == '__main__':
    main()