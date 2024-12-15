import torch
from torch.utils.data import DataLoader
from data_loaders_histo import IconColumnIterableDataset
import matplotlib.pyplot as plt
import os
import numpy as np
from os.path import join
import glob
import re
import random


height_level = 30

def plot_histograms(train_files, num_samples=500000, output_dir="histograms", height_level = height_level):
    # Create a dataset using the train files
    train_dataset = IconColumnIterableDataset(train_files)
    
    # Create a data loader for the train dataset
    data_loader = DataLoader(train_dataset, batch_size=2000, num_workers=0)
    
    flux_values = []
    
    # Collect flux values from the train dataset
    for i, (x3d, x2d, y) in enumerate(data_loader):
        
        # Extract the top 3 uppermost levels of flux values
        # top_flux = y[:, -22:, :]
        top_flux = y[:, height_level :]

        flux_values.append(top_flux)
        
        # Print a message after every 10000 samples are processed
        if (i + 1) * data_loader.batch_size % 10000 == 0:
            print(f"Processed {(i + 1) * data_loader.batch_size} samples")
        
        if (i + 1) * data_loader.batch_size >= num_samples:
            break
    
    flux_values = torch.cat(flux_values, dim=0)  # Shape: [total_samples, 3, num_fluxes]
    
    
    
    
    
    # Reshape flux_values to combine samples and levels
    flux_values = flux_values.view(-1, flux_values.size(-1))  # Shape: [total_samples * 3, num_fluxes]
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot histograms for each flux value at the top 3 levels
    num_fluxes = flux_values.size(1)
    
    flux_labels = ['LW Up', 'LW Down', 'SW Up', 'SW Down']
    colors = ['blue', 'green', 'red', 'purple']
    bins = [500, 500, 500, 500]  # Specify the number of bins for each flux
    
    for i in range(num_fluxes):
        flux_i = flux_values[:, i]
        
        # Remove zero and NaN values from the flux data
        non_zero_mask = (flux_i != 0)
        non_nan_mask = ~torch.isnan(flux_i)
        valid_mask = non_zero_mask & non_nan_mask
        flux_i = flux_i[valid_mask]
        
        if flux_i.numel() == 0:
            print(f"No valid samples for {flux_labels[i]} flux. Skipping plotting.")
            continue
        
        flux_i_numpy = flux_i.numpy()
        
        plt.figure(figsize=(8, 6))
        plt.hist(flux_i_numpy, bins=bins[i], alpha=0.7, color=colors[i], edgecolor='black')
        plt.title(f'Histogram of {flux_labels[i]} Flux Values (Level {height_level}, Non-Zero)')
        plt.xlabel('Flux Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        
        histogram_path = os.path.join(output_dir, f'histogram_{flux_labels[i].replace(" ", "_").lower()}_nonzero_{height_level}.png')
        plt.savefig(histogram_path)
        plt.close()
        print(f'Histogram for {flux_labels[i]} flux (non-zero, {height_level} levels) saved as {histogram_path}')

    

# Set the random seed and RNG states
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Save RNG states before model initialization
torch_rng_state = torch.get_rng_state()
np_rng_state = np.random.get_state()
random_rng_state = random.getstate()

# Specify the dataset path
dataset_path = "/mydata/deepcloud/salman/dataset/h5_data_all_chuncked"

# Get the list of filenames in the dataset path
filenames = glob.glob(join(dataset_path, '*.h5'))

# Extract the time indices from the filenames
time_indices = [float(re.search(r'\_time_(.*?)\.h5', f).group(1)) for f in filenames]

# Sort the filenames based on the time indices
sorted_files = [x for _, x in sorted(zip(time_indices, filenames))]

# Specify the train files
train_files = sorted_files[200:2000]

plot_histograms(train_files, output_dir="/mydata/deepcloud/yves/results_git/data_histograms")
