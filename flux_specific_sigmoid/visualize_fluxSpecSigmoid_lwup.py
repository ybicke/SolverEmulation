import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import re
#from data_loaders_new import IconColumnIterableDataset
from torch.utils.data import Dataset, DataLoader


import torch
import math


def load_gaussian_parameters(file_path):
    data = np.load(file_path)
    amplitudes = data['amplitudes']
    means = data['means']
    stddevs = data['stddevs']
    fitted_gaussians = list(zip(amplitudes, means, stddevs))
    print(f"Fitted Gaussian parameters loaded from {file_path}")
    return fitted_gaussians

def construct_gaussian_params_by_height(fitted_gaussians, height_start=70, height_end=69):
    gaussian_params_by_height = {}
    accumulated_gaussians = []
    index = 0

    for h in range(height_start, height_end - 1, -1):
        accumulated_gaussians.append(fitted_gaussians[index])
        gaussian_params_by_height[h] = accumulated_gaussians.copy()
        index += 1

    return gaussian_params_by_height


def compute_composite_sigmoid(x_flux, amplitudes, means, steepnesses):
    exponents = -steepnesses * (x_flux.unsqueeze(-1) - means)
    exponents = torch.clamp(exponents, min=-30, max=30)
    sigmoid_components = amplitudes / (1 + torch.exp(exponents))
    composite_sigmoid = sigmoid_components.sum(dim=-1)
    return composite_sigmoid


def normalize_amplitudes(composite_sigmoid, amplitudes):
    amplitude_sum = amplitudes.sum()
    normalized_sigmoid = composite_sigmoid / amplitude_sum
    return normalized_sigmoid



def gaussian_cdf(x, mean, std):
    return 0.5 * (1 + torch.erf((x.unsqueeze(-1) - mean) / (std * math.sqrt(2))))

def compute_composite_sigmoid_cdf(x_flux, amplitudes, means, stddevs):
    # Compute CDF components
    cdf_components = gaussian_cdf(x_flux, means, stddevs)
    
    # Compute weighted sum of CDF components
    weighted_cdf_components = cdf_components * amplitudes
    composite_sigmoid = weighted_cdf_components.sum(dim=-1)
    
    # Normalize the composite sigmoid
    amplitude_sum = amplitudes.sum()
    normalized_composite_sigmoid = composite_sigmoid / amplitude_sum
    
    return normalized_composite_sigmoid



def main():
    # Load the Gaussian parameters from the file
    output_dir = '/mydata/deepcloud/yves/results_git/data_histograms/sigmoid_plotting_lwUp/'
    os.makedirs(output_dir, exist_ok=True)
    
    fitted_gaussians_file = '/mydata/deepcloud/yves/results_git/data_histograms/histogram_fit_lwUp/fitted_gaussians_lwUp.npz'
    fitted_gaussians = load_gaussian_parameters(fitted_gaussians_file)

    # Construct the dictionary mapping height levels to Gaussian parameters
    # gaussian_params_by_height = construct_gaussian_params_by_height(fitted_gaussians)

    # Select a specific height level for simulation
    height_level = 65

    # Get the Gaussian parameters for the selected height level
    params = gaussian_params_by_height[height_level]
    amplitudes = torch.tensor([amp for amp, _, _ in params]).unsqueeze(0)  # Shape: [1, n_params]
    means = torch.tensor([mean for _, mean, _ in params]).unsqueeze(0)  # Shape: [1, n_params]
    stddevs = torch.tensor([stddev for _, _, stddev in params]).unsqueeze(0)  # Shape: [1, n_params]
    steepnesses = 3/ stddevs
    


    # Define the range of x values based on the means and stddevs of the selected Gaussians
    x_min = min([mean - 2 * stddev for amp, mean, stddev in params])
    x_max = max([mean + 2 * stddev for amp, mean, stddev in params])
    x_values = torch.linspace(x_min, x_max, 1000)
    
    # Compute the composite sigmoid
    composite_sigmoid = compute_composite_sigmoid(x_values, amplitudes, means, steepnesses)

    # Normalize the composite sigmoid using amplitude normalization
    normalized_sigmoid_amplitudes = normalize_amplitudes(composite_sigmoid, amplitudes)
    
    # Compute the composite sigmoid using CDF
    composite_sigmoid_cdf = compute_composite_sigmoid_cdf(x_values, amplitudes, means, stddevs)

    # Plot the composite sigmoid
    plt.figure(figsize=(10, 6))
    plt.plot(x_values.numpy(), composite_sigmoid.squeeze().numpy(), label='Composite Sigmoid')
    plt.xlabel('Input')
    plt.ylabel('Output')    
    plt.title(f'Composite Sigmoid Function (Height Level: {height_level})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'composite_sigmoid_plot_clamp30.png'))
    plt.show()
    
    
    
    # Plot the composite sigmoid using CDF
    plt.figure(figsize=(10, 6))
    plt.plot(x_values.numpy(), composite_sigmoid_cdf.squeeze().numpy(), label='Composite Sigmoid (CDF)')
    plt.xlabel('Input')
    plt.ylabel('Output')    
    plt.title(f'Composite Sigmoid Function using CDF (Height Level: {height_level})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'composite_sigmoid_cdf_plot_clamp30.png'))
    plt.show()



    # Plot the amplitude normalized sigmoid
    plt.figure(figsize=(10, 6))
    plt.plot(x_values.numpy(), normalized_sigmoid_amplitudes.squeeze().numpy(), label='Amplitude Normalized')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.title(f'Amplitude Normalized Sigmoid (Height Level: {height_level})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'amplitude_normalized_sigmoid_clamp30.png'))
    plt.show()

if __name__ == '__main__':
    main()