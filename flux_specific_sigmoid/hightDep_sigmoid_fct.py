import torch
import torch.nn as nn
import numpy as np


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



def load_gaussian_parameters(file_path):
    data = np.load(file_path)
    amplitudes = data['amplitudes']
    means = data['means']
    stddevs = data['stddevs']
    fitted_gaussians = list(zip(amplitudes, means, stddevs))
    print(f"Fitted Gaussian parameters loaded from {file_path}")
    return fitted_gaussians



def construct_gaussian_params_by_height(fitted_gaussians, height_start=70, height_end=64):
    """
    Constructs a dictionary mapping height levels to their corresponding Gaussian parameters.
    Args:
        fitted_gaussians (list of tuples): The list of Gaussian parameters (amplitude, mean, stddev).
        height_start (int): The starting height level.
        height_end (int): The ending height level.
    Returns:
        dict: A dictionary where keys are height levels and values are lists of Gaussian parameters.
    """
    gaussian_params_by_height = {}
    accumulated_gaussians = []
    index = 0

    for h in range(height_start, height_end - 1, -1):  # Work downwards from height_start to height_end
        # Add exactly one additional Gaussian parameter for the current level
        accumulated_gaussians.append(fitted_gaussians[index])
        gaussian_params_by_height[h] = accumulated_gaussians.copy()  # Use copy to prevent reference issues
        index += 1

    return gaussian_params_by_height



def verify_gaussian_params(gaussian_params_by_height):

    for h in sorted(gaussian_params_by_height.keys(), reverse=True):
        print(f"Height {h}:")
        for amplitude, mean, std in gaussian_params_by_height[h]:
            print(f"  Amplitude: {amplitude}, Mean: {mean}, StdDev: {std}")




class HeightDependentSigmoid(nn.Module):
    def __init__(self, gaussian_params_by_height, flux_index):
        super(HeightDependentSigmoid, self).__init__()
        self.gaussian_params_by_height = gaussian_params_by_height
        self.flux_index = flux_index  # Index of the flux to apply the sigmoid to

    def forward(self, x):
        batch_size, height_size, num_fluxes = x.shape
        y = x.clone()

        for h in range(height_size):
            physical_height = h + 1  # Direct mapping

            if physical_height == height_size:
                # Set the very top layer to zero for the specified flux
                y[:, h, self.flux_index] = 0

            elif physical_height in self.gaussian_params_by_height:
                params = self.gaussian_params_by_height[physical_height]
                composite_sigmoid = torch.zeros((batch_size), device=x.device, dtype=x.dtype)

                for amplitude, mean, stddev in params:
                    steepness = 3 / stddev  # Adjust as needed
                    exponent = -steepness * (x[:, h, self.flux_index] - mean)
                    exponent = torch.clamp(exponent, min=-100, max=100)
                    sigmoid_component = amplitude / (1 + torch.exp(exponent))
                    composite_sigmoid += sigmoid_component

                # Normalize the composite sigmoid
                max_value = composite_sigmoid.max()
                if max_value == 0:
                    max_value = 1  # Avoid division by zero
                composite_sigmoid /= max_value

                # Update the flux at height h
                y[:, h, self.flux_index] = composite_sigmoid
            else:
                # Apply standard sigmoid
                y[:, h, self.flux_index] = torch.sigmoid(x[:, h, self.flux_index])

        return y



def plot_height_dependent_sigmoids(gaussian_params_by_height, save_directory, height_sigmoid, flux_index=2):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Determine all physical heights to plot
    heights = sorted(gaussian_params_by_height.keys())
    height_size = max(heights) + 1  # Since physical heights start from 1

    for h_physical in heights:
        # Get the parameters for the current physical height to determine x-range
        params = gaussian_params_by_height[h_physical]
        means = [mean for (_, mean, _) in params]
        stddevs = [stddev for (_, _, stddev) in params]

        # Determine x-range based on means and stddevs
        x_min = min(mean - 3 * stddev for mean, stddev in zip(means, stddevs))
        x_max = max(mean + 3 * stddev for mean, stddev in zip(means, stddevs))
        x_min = max(x_min, 0)  # Ensure x_min is not negative
        x_range = np.linspace(x_min, x_max, 1000)

        # Prepare input tensor with shape [batch_size=num_points, height_size, num_fluxes]
        num_points = len(x_range)
        batch_size = num_points  # Process all x_range values in parallel
        num_fluxes = 4  # Adjust if necessary

        # Initialize input tensor with zeros
        input_tensor = torch.zeros((batch_size, height_size, num_fluxes), dtype=torch.float32)

        # Map physical height to tensor index
        h_tensor = h_physical - 1  # Since physical_height = h + 1, we have h = physical_height - 1

        # Set the 'LW Down' flux at tensor index h_tensor to x_range values
        input_tensor[:, h_tensor, flux_index] = torch.tensor(x_range, dtype=torch.float32)

        # Apply the HeightDependentSigmoid transformation
        with torch.no_grad():
            output_tensor = height_sigmoid(input_tensor)

        # Extract the output for 'LW Down' flux at tensor index h_tensor
        y_values = output_tensor[:, h_tensor, flux_index].numpy()

        # Plot the result
        plt.figure(figsize=(10, 6))
        plt.plot(x_range, y_values, label=f'Physical Height {h_physical}', color='black', linewidth=2)

        plt.title(f'Sigmoid Function at Physical Height {h_physical} for LW Down Flux')
        plt.xlabel('Flux Value (LW Down)')
        plt.ylabel('Sigmoid Output')
        plt.grid(True)
        plt.legend()

        # Save the plot
        file_name = f'sigmoid_height_{h_physical}_lw_down.png'
        save_path = os.path.join(save_directory, file_name)
        plt.savefig(save_path)
        plt.close()



if __name__ == "__main__":
    # Define parameters
    fitted_gaussians_file = '/mydata/deepcloud/yves/results_git/data_histograms/histogram_fit_lwDown/fitted_gaussians_lwDown.npz'
    save_directory = '/mydata/deepcloud/yves/results_git/data_histograms/sigmoid_plots'
    flux_index = 2  # Adjust if necessary
    height_start = 70
    height_end = 64

    fitted_gaussians = load_gaussian_parameters(fitted_gaussians_file)

    gaussian_params_by_height = construct_gaussian_params_by_height(fitted_gaussians, 
        height_start=height_start, 
        height_end=height_end)

    verify_gaussian_params(gaussian_params_by_height)

    height_sigmoid = HeightDependentSigmoid(
        gaussian_params_by_height, 
        flux_index = flux_index)

    # Plot the sigmoids
    plot_height_dependent_sigmoids(
        gaussian_params_by_height,
        save_directory,
        height_sigmoid,
        flux_index=flux_index
    )
