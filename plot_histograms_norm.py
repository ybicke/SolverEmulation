import torch
from torch.utils.data import DataLoader
from data_loaders_histo import IconColumnIterableDataset
import matplotlib.pyplot as plt
import os
from os.path import join
import numpy as np
import pickle




stats_file = join('/mydata/deepcloud/salman/dataset/h5_data_all_chuncked', 'normalizer_stats_per_feat.pickle')

# Load the stats file
with open(stats_file, 'rb') as f:
    stats = pickle.load(f)

# Print the keys of the stats dictionary
print("Keys in the stats file:", stats.keys())

# Print the shape and contents of each statistic
for key, value in stats.items():
    print(f"Key: {key}")
    print(f"Shape: {value.shape}")
    print(f"Contents: {value}")
    print("---")


def plot_histograms(data_loader, num_samples=2000, output_dir="histograms"):
    flux_values = []
    
    # Collect flux values from the dataset
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
    
    # Normalize and center the flux values
    mean = torch.nanmean(flux_values, dim=0)
    valid_mask = ~torch.isnan(flux_values)
    std = torch.sqrt(torch.nanmean((flux_values - mean)**2, dim=0))
    normalized_flux_values = (flux_values - mean) / std
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot histograms for each normalized and centered flux value at the uppermost level
    num_fluxes = normalized_flux_values.size(1)
    flux_labels = ['LW Up', 'LW Down', 'SW Up', 'SW Down']
    colors = ['blue', 'green', 'red', 'purple']
    bins = [50, 100, 50, 50]  # Specify the number of bins for each flux
    
    for i in range(num_fluxes):
        flux_i = normalized_flux_values[:, i].numpy()
        # Remove NaN values from the flux data
        flux_i = flux_i[~np.isnan(flux_i)]
        plt.figure(figsize=(8, 6))
        plt.hist(flux_i, bins=bins[i], alpha=0.7, color=colors[i], edgecolor='black')
        plt.title(f'Histogram of {flux_labels[i]} Flux Values (Uppermost Level, Normalized and Centered)')
        plt.xlabel('Normalized and Centered Flux Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        
        histogram_path = os.path.join(output_dir, f'histogram_{flux_labels[i].replace(" ", "_").lower()}_normalized_centered.png')
        plt.savefig(histogram_path)
        plt.close()
        print(f'Histogram for {flux_labels[i]} flux (normalized and centered) saved as {histogram_path}')
    
    # Access specific flux values for closer investigation
    lw_down_flux = normalized_flux_values[:, 1].numpy()
    print(f"LW Down flux values (normalized and centered): {lw_down_flux}")
    print(f"Standard deviation values: {std}")
    print(f"Flux values for a specific sample: {flux_values[0]}")
    print(f"Number of NaN values in flux_values: {torch.isnan(flux_values).sum().item()}")

    # You can perform further analysis or investigation on the lw_down_flux values here
    
    

dataset_path = "/mydata/deepcloud/salman/dataset/h5_data_all_chuncked"
filenames = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path) if file.endswith(".h5")]

dataset = IconColumnIterableDataset(filenames)
data_loader = DataLoader(dataset, batch_size=1000, num_workers=0)

plot_histograms(data_loader, output_dir="/mydata/deepcloud/yves/results_git/data_histograms")
