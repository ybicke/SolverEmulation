import torch
from torch.utils.data import DataLoader
from data_loaders_histo import IconColumnIterableDataset
import matplotlib.pyplot as plt
import os
import numpy as np
from os.path import join


def plot_histograms(train_files, num_samples=5000000, output_dir="histograms"):
    # Create a dataset using the train files
    train_dataset = IconColumnIterableDataset(train_files)
    
    # Create a data loader for the train dataset
    data_loader = DataLoader(train_dataset, batch_size=2000, num_workers=0)
    
    flux_values = []
    
    # Collect flux values from the train dataset
    for i, (x3d, x2d, y) in enumerate(data_loader):
        # Extract the uppermost level of flux values
        uppermost_flux = y[:, -1, :]  # Assumes the second last dimension represents height
        flux_values.append(uppermost_flux)
        
        # Print a message after every 500 samples are processed
        if (i + 1) * data_loader.batch_size % 10000 == 0:
            print(f"Processed {(i + 1) * data_loader.batch_size} samples")
        
        if (i + 1) * data_loader.batch_size >= num_samples:
            break
    
    flux_values = torch.cat(flux_values, dim=0)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot histograms for each flux value at the uppermost level
    num_fluxes = flux_values.size(1)
    flux_labels = ['LW Up', 'LW Down', 'SW Up', 'SW Down']
    colors = ['blue', 'green', 'red', 'purple']
    bins = [500, 500, 500, 500]  # Specify the number of bins for each flux
    
    for i in range(num_fluxes):
        flux_i = flux_values[:, i].numpy()
        flux_i = flux_i[~np.isnan(flux_i)]
        
        # Remove zero values from the flux data
        flux_i = flux_i[flux_i != 0]
        
        plt.figure(figsize=(8, 6))
        plt.hist(flux_i, bins=bins[i], alpha=0.7, color=colors[i], edgecolor='black')
        plt.title(f'Histogram of {flux_labels[i]} Flux Values (Uppermost Level, Non-zero)')
        plt.xlabel('Flux Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        
        histogram_path = os.path.join(output_dir, f'histogram_{flux_labels[i].replace(" ", "_").lower()}_standard_nonzero.png')
        plt.savefig(histogram_path)
        plt.close()
        print(f'Histogram for {flux_labels[i]} flux saved as {histogram_path}')
    
    # Access specific flux values for closer investigation
    lw_down_flux = flux_values[:, 1].numpy()
    print(f"LW Down flux values: {lw_down_flux}")
    # You can perform further analysis or investigation on the lw_down_flux values here

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

# Specify the train files
train_files = sorted_files[200:2000]

plot_histograms(train_files, output_dir="/mydata/deepcloud/yves/results_git/data_histograms")