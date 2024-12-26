import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from data_loaders_new import IconColumnIterableDataset
from torch.utils.data import Dataset, DataLoader


def load_gaussian_parameters(file_path):
    data = np.load(file_path)
    amplitudes = data['amplitudes']
    means = data['means']
    stddevs = data['stddevs']
    fitted_gaussians = list(zip(amplitudes, means, stddevs))
    print(f"Fitted Gaussian parameters loaded from {file_path}")
    return fitted_gaussians

def construct_gaussian_params_by_height(fitted_gaussians, height_start=70, height_end=64):
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

def main():
    # Load the Gaussian parameters from the file
    output_dir = '/mydata/deepcloud/yves/results_git/data_histograms/sigmoid_plotting/'
    os.makedirs(output_dir, exist_ok=True)

    fitted_gaussians_file = '/mydata/deepcloud/yves/results_git/data_histograms/histogram_fit_lwDown/fitted_gaussians_lwDown.npz'
    fitted_gaussians = load_gaussian_parameters(fitted_gaussians_file)

    # Construct the dictionary mapping height levels to Gaussian parameters
    gaussian_params_by_height = construct_gaussian_params_by_height(fitted_gaussians)

    # Loop over each height level in the dictionary
    for height_level in gaussian_params_by_height.keys():
        params = gaussian_params_by_height[height_level]
        amplitudes = torch.tensor([amp for amp, _, _ in params])    # Shape: [n_params]
        means = torch.tensor([mean for _, mean, _ in params])       # Shape: [n_params]
        stddevs = torch.tensor([stddev for _, _, stddev in params]) # Shape: [n_params]
        steepnesses = 5 / stddevs  # Adjust as needed

        # Compute overall mean and standard deviation for the current height level
        total_amplitude = amplitudes.sum()
        overall_mean = (amplitudes * means).sum() / total_amplitude
        variance = (amplitudes * ((means - overall_mean) ** 2 + stddevs ** 2)).sum() / total_amplitude
        overall_stddev = torch.sqrt(variance)

        # Shift and scale x-values to center at zero and match standard sigmoid scale
        x_values = torch.linspace(overall_mean - 6 * overall_stddev, overall_mean + 6 * overall_stddev, 1000)
        shifted_x_values = (x_values - overall_mean) / overall_stddev  # Now x ranges approximately from -6 to +6

        # Compute the composite sigmoid using original x_values
        composite_sigmoid = compute_composite_sigmoid(
            x_values, 
            amplitudes.unsqueeze(0), 
            means.unsqueeze(0), 
            steepnesses.unsqueeze(0)
        )

        # Compute standard sigmoid over the shifted x-values
        standard_sigmoid = 1 / (1 + torch.exp(-shifted_x_values))

        # Normalize the composite sigmoid for comparison
        composite_sigmoid_normalized = composite_sigmoid / composite_sigmoid.max()

        # Plot the composite sigmoid and standard sigmoid
        plt.figure(figsize=(10, 6))
        plt.plot(shifted_x_values.numpy(), composite_sigmoid_normalized.squeeze().numpy(), label='Composite Sigmoid')
        plt.plot(shifted_x_values.numpy(), standard_sigmoid.numpy(), label='Standard Sigmoid', linestyle='--')
        plt.xlabel('Input (Shifted and Scaled)')
        plt.ylabel('Output')
        plt.title(f'Composite vs. Standard Sigmoid (Height Level: {height_level})')
        plt.legend()
        plt.grid(True)

        # Save the plot with a filename that includes the height level
        plt.savefig(os.path.join(output_dir, f'composite_vs_standard_sigmoid_height_{height_level}.png'))
        plt.close()  # Close the figure to free up memory

    # Optionally, display plots for specific height levels
    for height_level in [68, 66, 64]:
        if height_level in gaussian_params_by_height:
            params = gaussian_params_by_height[height_level]
            amplitudes = torch.tensor([amp for amp, _, _ in params])    # Shape: [n_params]
            means = torch.tensor([mean for _, mean, _ in params])       # Shape: [n_params]
            stddevs = torch.tensor([stddev for _, _, stddev in params]) # Shape: [n_params]
            steepnesses = 5 / stddevs

            # Compute overall mean and standard deviation
            total_amplitude = amplitudes.sum()
            overall_mean = (amplitudes * means).sum() / total_amplitude
            variance = (amplitudes * ((means - overall_mean) ** 2 + stddevs ** 2)).sum() / total_amplitude
            overall_stddev = torch.sqrt(variance)

            # Shift and scale x-values
            x_values = torch.linspace(overall_mean - 6 * overall_stddev, overall_mean + 6 * overall_stddev, 1000)
            shifted_x_values = (x_values - overall_mean) / overall_stddev

            # Compute the composite sigmoid
            composite_sigmoid = compute_composite_sigmoid(
                x_values, 
                amplitudes.unsqueeze(0), 
                means.unsqueeze(0), 
                steepnesses.unsqueeze(0)
            )
            standard_sigmoid = 1 / (1 + torch.exp(-shifted_x_values))

            # Normalize the composite sigmoid for comparison
            composite_sigmoid_normalized = composite_sigmoid / composite_sigmoid.max()

            # Plot and display
            plt.figure(figsize=(10, 6))
            plt.plot(shifted_x_values.numpy(), composite_sigmoid_normalized.squeeze().numpy(), label='Composite Sigmoid')
            plt.plot(shifted_x_values.numpy(), standard_sigmoid.numpy(), label='Standard Sigmoid', linestyle='--')
            plt.xlabel('Input (Shifted and Scaled)')
            plt.ylabel('Output')
            plt.title(f'Composite vs. Standard Sigmoid (Height Level: {height_level})')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'composite_vs_standard_sigmoid_height_{height_level}.png'))
            plt.show()

if __name__ == '__main__':
    main()