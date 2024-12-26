import torch
from torch.utils.data import DataLoader
from data_loaders_histo import IconColumnIterableDataset
import matplotlib.pyplot as plt
import os
from os.path import join
import numpy as np
import glob
import re
import random

def plot_histograms(train_files, num_samples=1000000, output_dir="histograms"):
    # Create a dataset using the train files
    train_dataset = IconColumnIterableDataset(train_files)

    # Create a data loader for the train dataset
    data_loader = DataLoader(train_dataset, batch_size=2000, num_workers=0)

    flux_values = []

    # Collect flux values from the train dataset
    for i, (x3d, x2d, y) in enumerate(data_loader):
        # Extract the top 5 uppermost levels of flux values
        top_flux = y[:, -5:, :]  
        flux_values.append(top_flux)

        # Print a message after every 10000 samples are processed
        if (i + 1) * data_loader.batch_size % 10000 == 0:
            print(f"Processed {(i + 1) * data_loader.batch_size} samples")

        if (i + 1) * data_loader.batch_size >= num_samples:
            break

    flux_values = torch.cat(flux_values, dim=0)  # Shape: [total_samples, 5, 4]
    print(f"flux_values shape: {flux_values.shape}")  # Should output [total_samples, 5, 4]

    # Create a mask to filter out zero values only
    non_zero_mask = (flux_values != 0)

    # Apply the mask to get valid flux values only
    valid_flux_values = flux_values[non_zero_mask].view(-1, flux_values.size(-1))  # Shape: [total_valid_samples, 4]

    # Calculate mean and std across all valid values for each flux variable
    mean = torch.mean(valid_flux_values, dim=0)  # Shape: [4]
    std = torch.sqrt(torch.mean((valid_flux_values - mean) ** 2, dim=0))  # Shape: [4]
    print(f"mean: {mean}, std: {std}")  

    # Normalize flux_values using the computed mean and std, keeping zero entries as zero
    normalized_flux_values = torch.where(
        non_zero_mask,
        (flux_values - mean) / std,
        torch.tensor(0.0).to(flux_values.device)  # Keep zero entries as zero
    )

    # Reshape normalized_flux_values for histogram plotting
    normalized_flux_values = normalized_flux_values.view(-1, normalized_flux_values.size(-1))  # Shape: [total_samples * 5, 4]

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define labels, colors, and bins for the four flux variables
    flux_labels = ['LW Up', 'LW Down', 'SW Up', 'SW Down']
    colors = ['blue', 'green', 'red', 'purple']
    bins = [500, 500, 500, 500]

    for i in range(4):  # Only loop over the 4 flux variables
        flux_i = normalized_flux_values[:, i]

        # Remove zero values from the normalized flux data
        valid_flux_i = flux_i[flux_i != 0]

        if valid_flux_i.numel() == 0:
            print(f"No valid samples for {flux_labels[i]} flux. Skipping plotting.")
            continue

        flux_i_numpy = valid_flux_i.numpy()

        plt.figure(figsize=(8, 6))
        plt.hist(flux_i_numpy, bins=bins[i], alpha=0.7, color=colors[i], edgecolor='black')
        plt.title(f'Histogram of {flux_labels[i]} Flux Values (Top 5 Levels, Normalized)')
        plt.xlabel('Normalized Flux Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()

        histogram_path = os.path.join(output_dir, f'histogram_{flux_labels[i].replace(" ", "_").lower()}_normalized_top5.png')
        plt.savefig(histogram_path)
        plt.close()
        print(f'Histogram for {flux_labels[i]} flux (normalized, top 5 levels) saved as {histogram_path}')

# Set the random seed and RNG states
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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