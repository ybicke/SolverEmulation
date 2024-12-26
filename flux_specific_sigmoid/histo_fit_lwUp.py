# File: histo_fit_lwUp_single_layer.py

import torch
from torch.utils.data import DataLoader
from data_loaders_new import IconColumnIterableDataset
import matplotlib.pyplot as plt
import os
import numpy as np
import random
import glob  
import re

import sklearn
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import jensenshannon
from scipy.optimize import curve_fit




def extract_lw_up_flux_single_layer(train_files, flux_index, layer_index, num_samples):
    """
    Extract LW Up flux values from a single layer in the dataset.
    Parameters:
        train_files (list): List of training file paths.
        layer_index (int): Index of the height layer to extract data from.
        num_samples (int): Number of samples to extract.
    Returns:
        numpy.ndarray: Array of LW Up flux values.
    """
    train_dataset = IconColumnIterableDataset(train_files)
    data_loader = DataLoader(train_dataset, batch_size=2000, num_workers=0)

    lw_up_flux_values = []

    # Collect LW Up flux values from the train dataset
    for i, (x3d, x2d, y) in enumerate(data_loader):

        # Extract the LW Up flux from the specified layer
        lw_up_flux = y[:, layer_index, flux_index]  # Shape: [batch_size]
        lw_up_flux_values.append(lw_up_flux)

        # Print a message after every 100,000 samples are processed
        if (i + 1) * data_loader.batch_size % 100000 == 0:
            print(f"Processed {(i + 1) * data_loader.batch_size} samples")

        if (i + 1) * data_loader.batch_size >= num_samples:
            break

    # Concatenate all the flux values
    lw_up_flux_values = torch.cat(lw_up_flux_values, dim=0)

    # Remove NaN and zero values
    lw_up_flux_values = lw_up_flux_values[~torch.isnan(lw_up_flux_values)]
    lw_up_flux_values = lw_up_flux_values[lw_up_flux_values != 0]

    return lw_up_flux_values.numpy()



def create_histogram_single_layer(flux_values, bins=500, save_path='histogram_lwup_single_layer.npz'):
    """
    Create histogram data for LW Up flux values from a single layer and save it.
    Returns:
        tuple: Bin centers, counts, and bin edges of the histogram.
    """
    counts, bin_edges = np.histogram(flux_values, bins=bins, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # normalize the histogram
    bin_width = bin_edges[1] - bin_edges[0]
    normalized_counts = counts / (counts.sum() * bin_width)

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save histogram data
    np.savez(save_path, bin_centers=bin_centers, counts=counts, bin_edges=bin_edges)
    print(f"Histogram data saved to {save_path}")

    return bin_centers, normalized_counts, bin_edges



def load_histogram_data(load_path='histogram_lwup_single_layer.npz'):
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Histogram data file '{load_path}' not found. Please create it first.")
    # Load the histogram data from the file
    data = np.load(load_path)
    bin_centers = data['bin_centers']
    counts = data['counts']
    print(f"Histogram data loaded from {load_path}")
    return bin_centers, counts

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

def fit_gaussian(x, y):
    amplitude_guess = np.max(y) 
    mean_guess = x[np.argmax(y)]
    stddev_guess = (x[-1] - x[0]) / 10  # Adjusted for a reasonable initial guess
    p0 = [amplitude_guess, mean_guess, stddev_guess]

    try:
        popt, _ = curve_fit(gaussian, x, y, p0=p0)
        return popt  # amplitude, mean, stddev
    except RuntimeError:
        print("Gaussian fit did not converge")
        return None

# def iterative_gaussian_fitting(x, y, initial_means, initial_amplitudes, num_gaussians=5, distance_threshold=0.1):
#     """
#     Fit multiple Gaussians to the histogram data iteratively.

#     Parameters:
#         x (numpy.ndarray): Bin centers.
#         y (numpy.ndarray): Histogram counts.
#         initial_means (list): List of initial mean estimates for each Gaussian.
#         initial_amplitudes (list): List of initial amplitude estimates for each Gaussian.
#         num_gaussians (int): Number of Gaussians to fit.
#         max_iterations (int): Maximum number of iterations.
#         distance_threshold (float): Threshold for stopping criterion based on Jensen-Shannon distance.

#     Returns:
#         tuple: List of fitted Gaussians and the cumulative model.
#     """
#     y = y.astype(np.float64)  # Ensure y is float64
#     cumulative_model = np.zeros_like(y)
#     residuals = y.copy()
#     fitted_gaussians = []

#     for iteration in range(num_gaussians):
#         # Get initial guess for the current Gaussian
#         mean = initial_means[iteration]
#         amplitude = initial_amplitudes[iteration]
#         stddev = np.std(residuals)  # Use the standard deviation of residuals as an initial guess

#         # Fit Gaussian to residuals with initial guess
#         popt, _ = curve_fit(gaussian, x, residuals, p0=[amplitude, mean, stddev])
#         amplitude, mean, stddev = popt
#         print(f"Fitted Gaussian {iteration + 1}: Amplitude = {amplitude}, Mean = {mean}, Stddev = {stddev}")
#         fitted_gaussians.append((amplitude, mean, stddev))

#         gaussian_fit = gaussian(x, amplitude, mean, stddev)
#         cumulative_model += gaussian_fit

#         # Update residuals
#         residuals = y - cumulative_model
#         residuals[residuals < 0] = 0  # Ensure no negative values

#         # Calculate Jensen-Shannon distance
#         js_distance = jensenshannon(y, cumulative_model)
#         print(f"Iteration {iteration + 1}: JS Distance = {js_distance}")

#         # Check stopping criterion
#         if js_distance < distance_threshold or iteration + 1 >= num_gaussians:
#             print("Stopping criterion reached.")
#             break

#     return fitted_gaussians, cumulative_model


def fit_gaussian_mixture(bin_centers, counts, n_components=3):
    """
    Fit a Gaussian Mixture Model (GMM) to the histogram data.
    Parameters:
        bin_centers (numpy.ndarray): The centers of the histogram bins.
        counts (numpy.ndarray): The counts in each histogram bin.
        n_components (int): Number of Gaussian components to fit.
    Returns:
        tuple: Fine-grained x values, evaluated GMM densities, weights, means, variances.
    """
    # Normalize counts to get probabilities
    probabilities = normalized_counts / normalized_counts.sum()

    # Number of samples to generate (same as histogram)
    n_samples = 3000000  # Adjust as needed

    # Sample data points from the histogram according to the probabilities
    sampled_data = np.random.choice(bin_centers, size=n_samples, p=probabilities)
    X = sampled_data.reshape(-1, 1)

    # Fit Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(X)

    # Generate fine-grained x values for plotting the fitted mixture distribution
    x_fine = np.linspace(bin_centers.min(), bin_centers.max(), 1000).reshape(-1, 1)

    # Evaluate the fitted mixture densities on the fine-grained x values
    y_fine = np.exp(gmm.score_samples(x_fine))

    # Extract weights, means, and variances
    weights = gmm.weights_
    means = gmm.means_.flatten()
    variances = gmm.covariances_.reshape(n_components)

    return x_fine.ravel(), y_fine, weights, means, variances


def save_gaussian_parameters_npz(fitted_gaussians, file_path):
    amplitudes = np.array([amp for amp, _, _ in fitted_gaussians])
    means = np.array([mean for _, mean, _ in fitted_gaussians])
    stddevs = np.array([stddev for _, _, stddev in fitted_gaussians])
    np.savez(file_path, amplitudes=amplitudes, means=means, stddevs=stddevs)
    print(f"Fitted Gaussian parameters saved to {file_path}")

def load_gaussian_parameters_npz(file_path):
    data = np.load(file_path)
    amplitudes = data['amplitudes']
    means = data['means']
    stddevs = data['stddevs']
    fitted_gaussians = list(zip(amplitudes, means, stddevs))
    print(f"Fitted Gaussian parameters loaded from {file_path}")
    return fitted_gaussians

def sigmoid(x, amplitude, mean, steepness):
    return amplitude / (1 + np.exp(-steepness * (x - mean)))

def construct_composite_sigmoid(fitted_gaussians, path, num_gaussians_to_include=3):
    """
    Construct a composite sigmoid function based on the fitted Gaussians.
    Parameters:
        fitted_gaussians (list of tuples): List containing tuples of (amplitude, mean, stddev).
        path (str): File path to save the composite sigmoid plot.
        num_gaussians_to_include (int): Number of Gaussians to include.
    Returns:
        None
    """
    # Ensure there are enough Gaussians
    if len(fitted_gaussians) < num_gaussians_to_include:
        print(f"Only {len(fitted_gaussians)} Gaussians available. Using all of them.")
        num_gaussians_to_include = len(fitted_gaussians)

    # Select the desired number of Gaussians
    selected_gaussians = fitted_gaussians[:num_gaussians_to_include]

    # Define the range of x values based on the means and stddevs of the selected Gaussians
    x_min = min([mean - 3 * stddev for _, mean, stddev in selected_gaussians])
    x_max = max([mean + 3 * stddev for _, mean, stddev in selected_gaussians])
    x_values = np.linspace(x_min, x_max, 1000)

    # Initialize the composite sigmoid function
    composite_sigmoid = np.zeros_like(x_values)

    # Construct sigmoids for each selected Gaussian component
    for amplitude, mean, stddev in selected_gaussians:
        # Set steepness inversely proportional to stddev to have distinct inflection points
        steepness = 3 / stddev  # Adjust this multiplier as needed
        # Construct individual sigmoid centered at the mean
        sigmoid_component = sigmoid(x_values, amplitude, mean, steepness)
        # Add to composite sigmoid
        composite_sigmoid += sigmoid_component

    # Normalize the composite sigmoid to ensure it stays between 0 and 1
    composite_sigmoid /= np.max(composite_sigmoid)

    # Plot the composite sigmoid function
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, composite_sigmoid, label='Composite Sigmoid Function', color='green')

    # Plot vertical lines at the mean values to indicate inflection points
    for idx, (_, mean, _) in enumerate(selected_gaussians):
        if idx == 0:
            plt.axvline(x=mean, color='red', linestyle='--', label=f'Inflection Point at mean={mean:.4f}')
        else:
            plt.axvline(x=mean, color='red', linestyle='--')

    plt.title('Composite Sigmoid Function for LW Up Flux')
    plt.xlabel('Flux Value')
    plt.ylabel('Sigmoid Output')
    plt.grid(True)
    plt.legend()
    plt.savefig(path)  # Save the plot
    plt.show()


def main():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Update file paths for LW Up flux
    histogram_file = '/mydata/deepcloud/yves/results_git/data_histograms/histogram_fit_lwUp/histogram_lwup_single_layer.npz'
    histogram_plot = '/mydata/deepcloud/yves/results_git/data_histograms/histogram_fit_lwUp/histogram_lwup_single_layer_plot.png'
    fitted_gaussians_file = '/mydata/deepcloud/yves/results_git/data_histograms/histogram_fit_lwUp/fitted_gaussians_lwUp_plotting.npz'
    sigmoid_plot = '/mydata/deepcloud/yves/results_git/data_histograms/histogram_fit_lwUp/sigmoid_lwup_single_layer.png'

    flux_index = 0  # Adjust based on your data
    layer_index = 69  # Choose a valid layer index
    num_samples = 3000000  # Adjust as needed

    # Create Histogram
    if not os.path.exists(histogram_file):
        print(f"Histogram file '{histogram_file}' not found. Creating and saving histogram data.")
        dataset_path = "/mydata/deepcloud/salman/dataset/h5_data_all_chuncked"
        filenames = glob.glob(os.path.join(dataset_path, '*.h5'))
        time_indices = [float(re.search(r'\_time_(.*?)\.h5', f).group(1)) for f in filenames]
        sorted_files = [x for _, x in sorted(zip(time_indices, filenames))]
        train_files = sorted_files[200:2000]  # Adjust indices as needed

        # Extract LW Up flux values from a single layer
        lw_up_flux_values = extract_lw_up_flux_single_layer(train_files, flux_index, layer_index, num_samples)
        bin_centers, counts, bin_edges = create_histogram_single_layer(lw_up_flux_values, save_path=histogram_file)

        plt.figure(figsize=(8, 6))
        plt.plot(bin_centers, counts, label='Created Histogram upward flux')
        plt.title('LW Up Flux Histogram (Single Layer)')
        plt.xlabel('Flux Value')
        plt.ylabel('Count')
        plt.grid(True)
        plt.legend()
        plt.savefig(histogram_plot)
        plt.show()
        plt.close()
    else:
        print(f"Histogram file '{histogram_file}' found. Loading histogram data.")
        bin_centers, counts = load_histogram_data(load_path=histogram_file)

    counts = counts.astype(np.float64)

    if not os.path.exists(fitted_gaussians_file):
        n_components = 3  # Number of Gaussian components in the mixture
        x_fine, y_fine, weights, means, variances = fit_gaussian_mixture(bin_centers, counts, n_components)

        fitted_gaussians = []
        for i in range(n_components):
            weight = weights[i]
            mean = means[i]
            variance = variances[i]
            stddev = np.sqrt(variance)
            amplitude = weight / np.sqrt(2 * np.pi * variance)
            fitted_gaussians.append((amplitude, mean, stddev))
            print(f"Fitted Gaussian {i+1}: Amplitude = {amplitude:.4f}, Mean = {mean:.4f}, Stddev = {stddev:.4f}")

        # Plot the histogram with the fitted Gaussian mixture model
        plt.figure(figsize=(10, 6))
        plt.bar(bin_centers, counts, width=(bin_centers[1]-bin_centers[0]), edgecolor='black', linewidth=0.8, alpha=0.7, label='Histogram (Single Layer)')

        # Plot the individual Gaussian components
        x_plot = np.linspace(bin_centers.min(), bin_centers.max(), 1000)
        y_components = np.zeros_like(x_plot)
        for i in range(n_components):
            amplitude, mean, stddev = fitted_gaussians[i]
            y_component = amplitude * np.exp(-0.5 * ((x_plot - mean) / stddev) ** 2)
            y_components += y_component * counts.sum() * (bin_centers[1]-bin_centers[0])  # Scale to match histogram
            plt.plot(x_plot, y_component * counts.sum() * (bin_centers[1]-bin_centers[0]), linestyle='--', label=f'Gaussian Component {i+1}')

        # Plot the sum of Gaussian components
        plt.plot(x_plot, y_components, color='red', label='Sum of Gaussian Components')

        plt.title('True Histogram vs. Fitted Gaussian Mixture Model (Single Layer)')
        plt.xlabel('Flux Value')
        plt.ylabel('Count')
        plt.grid(True)
        plt.legend()
        plt.savefig('true_vs_fitted_gaussian_mixture_model_lwup_single_layer.png')
        plt.show()
        plt.close()

        # Save fitted Gaussian parameters
        save_gaussian_parameters_npz(fitted_gaussians, fitted_gaussians_file)
    else:
        # Load fitted Gaussian parameters from the file
        fitted_gaussians = load_gaussian_parameters_npz(fitted_gaussians_file)

    # Construct the composite sigmoid function
    construct_composite_sigmoid(fitted_gaussians, sigmoid_plot)


if __name__ == "__main__":
    main()
