
import sklearn
import torch
from torch.utils.data import DataLoader
from data_loaders_histo import IconColumnIterableDataset
import matplotlib.pyplot as plt
import os
import numpy as np
import random
from sklearn.mixture import GaussianMixture
import glob
import re

def extract_lw_down_flux_top_layers(train_files, num_samples=1000000):
    # Create a dataset using the train files
    train_dataset = IconColumnIterableDataset(train_files)

    # Create a data loader for the train dataset
    data_loader = DataLoader(train_dataset, batch_size=2000, num_workers=0)

    lw_down_flux_values = []

    # Collect LW Down flux values from the train dataset
    for i, (x3d, x2d, y) in enumerate(data_loader):
        # Extract the LW Down flux (assuming it's at index 1)
        top_lw_down_flux = y[:, -10:, 1]  # Adjust index if necessary

        # Flatten and collect flux values
        lw_down_flux_values.append(top_lw_down_flux.flatten())

        # Print a message after every 10000 samples are processed
        if (i + 1) * data_loader.batch_size % 100000 == 0:
            print(f"Processed {(i + 1) * data_loader.batch_size} samples")

        if (i + 1) * data_loader.batch_size >= num_samples:
            break

    # Concatenate all the flux values
    lw_down_flux_values = torch.cat(lw_down_flux_values, dim=0)

    # Remove zero and NaN values
    lw_down_flux_values = lw_down_flux_values[~torch.isnan(lw_down_flux_values)]
    lw_down_flux_values = lw_down_flux_values[lw_down_flux_values != 0]

    return lw_down_flux_values.numpy()

def create_histogram_top_layers(flux_values, bins=500, save_path='histogram_data.npz'):
    # Create a histogram
    counts, bin_edges = np.histogram(flux_values, bins=bins, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Save the histogram data to a file
    np.savez(save_path, bin_centers=bin_centers, counts=counts)
    print(f"Histogram data saved to {save_path}")

    # Plot the histogram
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, counts, label='Histogram (Top 10 Layers)')
    plt.title('LW Down Flux Histogram (Top 10 Layers)')
    plt.xlabel('Flux Value')
    plt.ylabel('Frequency')  # Changed to 'Frequency' as density=False
    plt.grid(True)
    plt.legend()
    plt.show()

    return bin_centers, counts

def load_histogram_data(load_path='histogram_data.npz'):
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Histogram data file '{load_path}' not found. Please create it first.")
    # Load the histogram data from the file
    data = np.load(load_path)
    bin_centers = data['bin_centers']
    counts = data['counts']
    print(f"Histogram data loaded from {load_path}")
    return bin_centers, counts

def histogram_to_data(bin_centers, counts):
    """
    Reconstruct raw data from histogram counts and bin centers.
    """
    data = np.repeat(bin_centers, counts.astype(int))
    return data

def select_gmm_components(data, max_components=15):
    """
    Select the optimal number of GMM components using BIC.
    """
    bic = []
    aic = []
    n_components_range = range(1, max_components + 1)
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        gmm.fit(data.reshape(-1, 1))
        bic.append(gmm.bic(data.reshape(-1, 1)))
        aic.append(gmm.aic(data.reshape(-1, 1)))

    # Plot BIC and AIC
    plt.figure(figsize=(8, 6))
    plt.plot(n_components_range, bic, label='BIC', marker='o')
    plt.plot(n_components_range, aic, label='AIC', marker='x')
    plt.xlabel('Number of Components')
    plt.ylabel('Information Criterion')
    plt.title('BIC and AIC for GMM')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Select the number of components with the lowest BIC
    optimal_n = n_components_range[np.argmin(bic)]
    print(f"Optimal number of components according to BIC: {optimal_n}")
    return optimal_n

def fit_gmm(data, n_components):
    """
    Fit a Gaussian Mixture Model to the data.
    """
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(data.reshape(-1, 1))
    return gmm

def extract_gmm_parameters(gmm):
    """
    Extract and sort GMM parameters.
    """
    amplitudes = gmm.weights_
    means = gmm.means_.flatten()
    stddevs = np.sqrt(gmm.covariances_.flatten())

    # Sort by mean
    sorted_indices = np.argsort(means)
    amplitudes = amplitudes[sorted_indices]
    means = means[sorted_indices]
    stddevs = stddevs[sorted_indices]

    return amplitudes, means, stddevs

def gaussian(x, mean, stddev):
    """
    Gaussian function normalized to unit amplitude.
    """
    return (1 / (stddev * np.sqrt(2 * np.pi))) * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

def plot_gmm_fit(bin_centers, counts, gmm, amplitudes, means, stddevs):
    """
    Plot the histogram data and the fitted GMM.
    """
    plt.figure(figsize=(10, 6))
    # Plot histogram as a bar plot
    plt.bar(bin_centers, counts, width=(bin_centers[1] - bin_centers[0]), alpha=0.6, color='gray', edgecolor='black', label='Histogram Data')

    # Plot GMM overall fit
    x_fit = np.linspace(bin_centers.min(), bin_centers.max(), 1000).reshape(-1, 1)
    gmm_pdf = np.exp(gmm.score_samples(x_fit)) * len(bin_centers) * (bin_centers[1] - bin_centers[0])
    plt.plot(x_fit, gmm_pdf, color='red', label='GMM Fit')

    # Plot individual Gaussian components
    for i, (amp, mean, std) in enumerate(zip(amplitudes, means, stddevs), 1):
        # Scale the Gaussian to match histogram counts
        gaussian_scaled = amp * gaussian(x_fit.flatten(), mean, std)
        plt.plot(x_fit.flatten(), gaussian_scaled, '--', label=f'Gaussian {i}: μ={mean:.6f}, σ={std:.6f}')

    plt.title('Gaussian Mixture Model Fit')
    plt.xlabel('Flux Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

def construct_composite_sigmoid_gmm(amplitudes, means, stddevs):
    """
    Construct and plot a composite sigmoid function based on GMM-fitted Gaussians.
    """
    # Define the range of x values
    x_min = min(means - 3 * stddevs)
    x_max = max(means + 3 * stddevs)
    x_values = np.linspace(x_min, x_max, 1000)

    # Initialize the composite sigmoid function
    composite_sigmoid = np.zeros_like(x_values)

    for amplitude, mean, stddev in zip(amplitudes, means, stddevs):
        # Set steepness inversely proportional to stddev
        steepness = 1 / (stddev + 1e-6)  # Add small value to prevent division by zero
        # Construct individual sigmoid
        sigmoid_component = amplitude / (1 + np.exp(-steepness * (x_values - mean)))
        # Add to composite
        composite_sigmoid += sigmoid_component

    # Plot the composite sigmoid function
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, composite_sigmoid, label='Composite Sigmoid Function', color='green')
    plt.title('Composite Sigmoid Function Based on GMM Fitted Gaussians')
    plt.xlabel('Flux Value')
    plt.ylabel('Sigmoid Output')
    plt.grid(True)
    plt.legend()
    plt.show()

def main_gmm_workflow():
    # Set random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Paths for the saved histogram file and plot
    histogram_file = '/mydata/deepcloud/yves/results_git/data_histograms/histogram_fit/histogram_GMM_top10_1mio.npz'
    histogram_plot = '/mydata/deepcloud/yves/results_git/data_histograms/histogram_fit/loaded_histogram_GMM_top10_1mio.png'

    # Check if the histogram file exists
    if os.path.exists(histogram_file):
        print(f"Histogram file '{histogram_file}' found. Loading histogram data.")
        bin_centers, counts = load_histogram_data(load_path=histogram_file)
    else:
        print(f"Histogram file '{histogram_file}' not found. Creating and saving histogram data.")

        # Specify dataset path and filenames
        dataset_path = "/mydata/deepcloud/salman/dataset/h5_data_all_chuncked"
        filenames = glob.glob(os.path.join(dataset_path, '*.h5'))

        # Extract time indices and sort files by time
        time_indices = [float(re.search(r'\_time_(.*?)\.h5', f).group(1)) for f in filenames]
        sorted_files = [x for _, x in sorted(zip(time_indices, filenames))]

        # Define training files
        train_files = sorted_files[200:2000]

        # Extract LW Down flux values from the top 10 layers
        lw_down_flux_values = extract_lw_down_flux_top_layers(train_files, num_samples=1000000)

        # Create histogram and save it
        bin_centers, counts = create_histogram_top_layers(lw_down_flux_values, save_path=histogram_file)

    # Plot the histogram for verification (as bar plot)
    plt.figure(figsize=(8, 6))
    plt.bar(bin_centers, counts, width=(bin_centers[1] - bin_centers[0]), alpha=0.6, color='gray', edgecolor='black', label='Histogram (Top 10 Layers)')
    plt.title('Loaded LW Down Flux Histogram (Top 10 Layers)')
    plt.xlabel('Flux Value')
    plt.ylabel('Frequency')  # Changed to 'Frequency' as density=False
    plt.grid(True)
    plt.legend()
    plt.savefig(histogram_plot)  # Save the plot
    plt.show()

    # Reconstruct raw data from histogram
    data = histogram_to_data(bin_centers, counts)
    print(f"Reconstructed raw data has {len(data)} points.")

    # Select the optimal number of GMM components
    optimal_n = select_gmm_components(data, max_components=15)

    # Fit GMM with the optimal number of components
    gmm = fit_gmm(data, n_components=optimal_n)

    # Extract GMM parameters
    amplitudes, means, stddevs = extract_gmm_parameters(gmm)

    # Print GMM parameters
    for i, (amp, mean, std) in enumerate(zip(amplitudes, means, stddevs), 1):
        print(f"Gaussian {i}: Amplitude = {amp:.2f}, Mean = {mean:.6f}, Stddev = {std:.6f}")

    # Plot GMM fit over the histogram
    plot_gmm_fit(bin_centers, counts, gmm, amplitudes, means, stddevs)

    # Construct the composite sigmoid function based on GMM parameters
    construct_composite_sigmoid_gmm(amplitudes, means, stddevs)

if __name__ == "__main__":
    main_gmm_workflow()