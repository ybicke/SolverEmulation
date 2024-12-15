import torch
from torch.utils.data import DataLoader
from data_loaders_histo import IconColumnIterableDataset
import matplotlib.pyplot as plt
import os
import numpy as np
import random
from scipy.optimize import curve_fit
from scipy.spatial.distance import jensenshannon
import glob  
import re    

def extract_lw_down_flux_top_layers(train_files, num_samples=1000000):
    
    train_dataset = IconColumnIterableDataset(train_files)
    data_loader = DataLoader(train_dataset, batch_size=2000, num_workers=0)
    
    lw_down_flux_values = []
    
    # Collect LW Down flux values from the train dataset
    for i, (x3d, x2d, y) in enumerate(data_loader):
        
        # Extract the LW Down flux 
        top_lw_down_flux = y[:, -6:, 1]  
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

    counts, bin_edges = np.histogram(flux_values, bins=bins, density=False) # here a density is created
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    np.savez(save_path, bin_centers=bin_centers, counts=counts)
    print(f"Histogram data saved to {save_path}")

    plt.figure(figsize=(8, 6))
    # plt.hist(flux_values, bins=bins, alpha=0.7, color='blue', edgecolor='black')  # Use plt.hist directly
    plt.plot(bin_centers, counts, label='Histogram (Top 10 Layers)')
    plt.title('LW Down Flux Histogram (Top 10 Layers)')
    plt.xlabel('Flux Value')
    plt.ylabel('Density')
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


def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))


def fit_gaussian(x, y):
    amplitude_guess = np.max(y) 
    mean_guess = x[np.argmax(y)]
    stddev_guess = (x[-1] - x[0]) / 1000 
    p0 = [amplitude_guess, mean_guess, stddev_guess]
    
    try:
        popt, _ = curve_fit(gaussian, x, y, p0=p0)
        return popt  # amplitude, mean, stddev
    except RuntimeError:
        print("Gaussian fit did not converge")
        return None


def iterative_gaussian_fitting(x, y, max_iterations=9, distance_threshold=1e-4):

    y = y.astype(np.float64)  # Ensure y is float64
    cumulative_model = np.zeros_like(y)
    residuals = y.copy()
    fitted_gaussians = []
    prev_distance = np.inf

    for iteration in range(max_iterations):
        
        # Fit Gaussian to residuals
        popt = fit_gaussian(x, residuals)
        if popt is None:
            break
        amplitude, mean, stddev = popt
        print(f"Fitted Gaussian: Amplitude = {amplitude}, Mean = {mean}, Stddev = {stddev}")
        fitted_gaussians.append((amplitude, mean, stddev))

        gaussian_fit = gaussian(x, amplitude, mean, stddev)
        cumulative_model += gaussian_fit

        # Update residuals
        residuals = y - cumulative_model
        residuals[residuals < 0] = 0  # Ensure no negative values

        # Calculate Jensen-Shannon distance
        js_distance = jensenshannon(y, cumulative_model)
        print(f"Iteration {iteration + 1}: JS Distance = {js_distance}")

        # Check stopping criterion
        if prev_distance - js_distance < distance_threshold:
            print("Stopping criterion reached.")
            break
        prev_distance = js_distance
        
    if not fitted_gaussians:
        print("No Gaussians were successfully fitted.")

    return fitted_gaussians, cumulative_model


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




def construct_composite_sigmoid(fitted_gaussians, path, num_gaussians_to_include=6):
    """
    Construct a composite sigmoid function with distinct inflection points based on selected fitted Gaussians.

    Parameters:
        fitted_gaussians (list of tuples): List containing tuples of (amplitude, mean, stddev).
        path (str): File path to save the composite sigmoid plot.
        num_gaussians_to_include (int): Number of Gaussians to use in constructing the sigmoid (e.g., first 1, 2, 3).
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
        steepness = 3 / stddev  # Adjust this multiplier as needed to emphasize the inflection points
        # Construct individual sigmoid centered at the mean
        exponents = -steepness * (x_values - mean)
        exponents = np.clip(exponents, -20, 20)  # Clamp the exponents
        sigmoid_component = amplitude / (1 + np.exp(exponents))
        # Add to composite sigmoid
        composite_sigmoid += sigmoid_component

    # Normalize the composite sigmoid to ensure it stays between 0 and 1
    composite_sigmoid /= np.max(composite_sigmoid)

    # Plot the composite sigmoid function
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, composite_sigmoid, label='Composite Sigmoid Function', color='green')

    # Plot vertical lines at the mean values to indicate inflection points
    #for _, mean, _ in selected_gaussians:
        #plt.scatter(mean, color='red', marker='o', label=f'Inflection Point at mean={mean:.4f}')
        #plt.axvline(x=mean, color='red', linestyle='--', label=f'Inflection Point at mean={mean:.4f}')

    plt.title('Composite Sigmoid Function with Clear Inflection Points at Selected Means')
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

    histogram_file = '/mydata/deepcloud/yves/results_git/data_histograms/histogram_fit_lwDown/histogram_fit_top10_1mio.npz'
    histogram_plot = '/mydata/deepcloud/yves/results_git/data_histograms/histogram_fit_lwDown/loaded_histogram_top10_1mio'
    fitted_gaussians_file = '/mydata/deepcloud/yves/results_git/data_histograms/histogram_fit_lwDown/fitted_gaussians_lwDown.npz'
    sigmoid_plot = '/mydata/deepcloud/yves/results_git/data_histograms/histogram_fit_lwDown/sigmoid_clip.png'


    if not os.path.exists(fitted_gaussians_file):
        if os.path.exists(histogram_file):
            print(f"Histogram file '{histogram_file}' found. Loading histogram data.")
            bin_centers, counts, bin_edges = load_histogram_data(load_path=histogram_file)
        else:
            print(f"Histogram file '{histogram_file}' not found. Creating and saving histogram data.")
            dataset_path = "/mydata/deepcloud/salman/dataset/h5_data_all_chuncked"
            filenames = glob.glob(os.path.join(dataset_path, '*.h5'))
            time_indices = [float(re.search(r'\_time_(.*?)\.h5', f).group(1)) for f in filenames]
            sorted_files = [x for _, x in sorted(zip(time_indices, filenames))]
            train_files = sorted_files[200:2000]
            lw_down_flux_values = extract_lw_down_flux_top_layers(train_files)
            bin_centers, counts, bin_edges = create_histogram_top_layers(lw_down_flux_values, save_path=histogram_file)

        plt.figure(figsize=(8, 6))
        plt.bar(bin_centers, counts, width=(bin_edges[1]-bin_edges[0]), edgecolor='black', linewidth=0.8, alpha=0.7, label='Histogram (Top 10 Layers)')
        plt.title('Loaded LW Down Flux Histogram (Top 10 Layers)')
        plt.xlabel('Flux Value')
        plt.ylabel('Density')
        plt.grid(True)
        plt.legend()
        plt.savefig(histogram_plot)
        plt.show()

        counts = counts.astype(np.float64)
        fitted_gaussians, cumulative_model = iterative_gaussian_fitting(bin_centers, counts)

        # Plot the histogram with the fitted individual Gaussians and cumulative model
        plt.figure(figsize=(10, 6))
        plt.bar(bin_centers, counts, width=(bin_edges[1]-bin_edges[0]), edgecolor='black', linewidth=0.8, alpha=0.7, label='Histogram (Top 10 Layers)')

        # Plot individual Gaussians
        for i, (amplitude, mean, stddev) in enumerate(fitted_gaussians):
            gaussian_curve = gaussian(bin_centers, amplitude, mean, stddev)
            plt.plot(bin_centers, gaussian_curve, linestyle='--', label=f'Fitted Gaussian {i+1}')

        # Plot cumulative model
        plt.plot(bin_centers, cumulative_model, label='Cumulative Gaussian Model', linestyle='-', color='red', linewidth=2)

        plt.title('Histogram with Fitted Gaussians and Cumulative Model (Top 10 Layers)')
        plt.xlabel('Flux Value')
        plt.ylabel('Density')
        plt.grid(True)
        plt.legend()
        plt.savefig('histogram_with_fitted_gaussians_lwdown_top_layers.png')
        plt.show()

        # Save fitted Gaussian parameters in the same directory as sigmoid_plot
        save_gaussian_parameters_npz(fitted_gaussians, fitted_gaussians_file)

    else:
        # Load fitted Gaussian parameters from the file
        fitted_gaussians = load_gaussian_parameters_npz(fitted_gaussians_file)

    # Construct the composite sigmoid function using the loaded or newly computed parameters
    construct_composite_sigmoid(fitted_gaussians, sigmoid_plot)

if __name__ == "__main__":
    main()
