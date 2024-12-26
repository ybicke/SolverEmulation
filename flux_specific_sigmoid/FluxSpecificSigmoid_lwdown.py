import torch
import torch.nn as nn
import numpy as np


import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np
from os.path import join
import glob
import re
import random
import math



# TODO: put in an utils file 
def load_gaussian_parameters(file_path):
    data = np.load(file_path)
    amplitudes = data['amplitudes']
    means = data['means']
    stddevs = data['stddevs']
    fitted_gaussians = list(zip(amplitudes, means, stddevs))
    print(f"Fitted Gaussian parameters loaded from {file_path}")
    return fitted_gaussians


# TODO: Could be integrated in the fitted gaussian process or put into the utils file
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





class HeightDependentSigmoid(nn.Module):
    def __init__(self, gaussian_params_by_height, flux_index):
        super(HeightDependentSigmoid, self).__init__()
        self.flux_index = flux_index

        # Prepare parameters for vectorized operations
        self.setup_parameters(gaussian_params_by_height)

    def setup_parameters(self, gaussian_params_by_height):

        # Create tensors to store parameters per height
        self.heights = sorted(gaussian_params_by_height.keys())
        self.amplitudes = []
        self.means = []
        self.stddevs = []
        self.steepnesses = []

        for h in self.heights:
            params = gaussian_params_by_height[h]
            amplitudes = torch.tensor([amp for amp, _, _ in params], dtype=torch.float32)
            means = torch.tensor([mean for _, mean, _ in params], dtype=torch.float32)
            stddevs = torch.tensor([stddev for _, _, stddev in params], dtype=torch.float32)
            epsilon = 1e-8
            steepnesses = 1 / (stddevs + epsilon) # Calculate steepnesses based on standard deviations

            self.amplitudes.append(amplitudes)
            self.means.append(means)
            self.steepnesses.append(steepnesses)
            self.stddevs.append(stddevs)


    def compute_composite_sigmoid(self, x, amplitudes, means, steepnesses):
        exponents = -steepnesses * (x.unsqueeze(1) - means)
        exponents = torch.clamp(exponents, min=-30, max=30)  # Clamp exponents to prevent overflow
        sigmoid_components = amplitudes / (1 + torch.exp(exponents))  # Compute sigmoid components
        composite_sigmoid = sigmoid_components.sum(dim=1)

        # Normalize the composite sigmoid by dividing it by the sum of the amplitudes
        amplitude_sum = amplitudes.sum()
        normalized_composite_sigmoid = composite_sigmoid / amplitude_sum

        return normalized_composite_sigmoid

    def gaussian_cdf(self, x, mean, std):
        return 0.5 * (1 + torch.erf((x.unsqueeze(-1) - mean) / (std * math.sqrt(2))))

    def compute_composite_sigmoid_cdf(self, x_flux, amplitudes, means, stddevs):
        # Compute CDF components
        cdf_components = self.gaussian_cdf(x_flux, means, stddevs)

        # Compute weighted sum of CDF components
        weighted_cdf_components = cdf_components * amplitudes
        composite_sigmoid = weighted_cdf_components.sum(dim=-1)

        # Normalize the composite sigmoid
        amplitude_sum = amplitudes.sum()
        normalized_composite_sigmoid = composite_sigmoid / amplitude_sum

        return normalized_composite_sigmoid


    def forward(self, x):
        batch_size, height_size, num_fluxes = x.shape
        device = x.device
        dtype = x.dtype

        modified_flux = torch.zeros(batch_size, height_size, device=device, dtype=dtype)

        for idx in range(height_size): # idx from 0 to 70
            phys_idx = idx + 1  

            if phys_idx in self.heights:                  
                # Get parameters for specific height index
                height_idx = self.heights.index(phys_idx)
                amplitudes = self.amplitudes[height_idx].to(device=device, dtype=dtype)
                means = self.means[height_idx].to(device=device, dtype=dtype)
                steepnesses = self.steepnesses[height_idx].to(device=device, dtype=dtype)
                stddevs = self.stddevs[height_idx].to(device=device, dtype=dtype)  # Define stddevs here


                # Extract the flux values at this height.
                x_h = x[:, idx, self.flux_index]  

                # Compute the custom sigmoid using the parameters per height
                #composite_sigmoid = self.compute_composite_sigmoid(x_h, amplitudes, means, steepnesses)
                composite_sigmoid = self.compute_composite_sigmoid_cdf(x_h, amplitudes, means, stddevs)
                modified_flux[:, idx] = composite_sigmoid 

            else:
                # Apply standard sigmoid for the remaining height layers
                # x_h = x[:, idx, self.flux_index]  
                modified_flux[:, idx] = torch.sigmoid(x[:, idx, self.flux_index])

        # Expand modified_flux to match the shape of x
        modified_flux = modified_flux.unsqueeze(-1)

        # Create mask for specified flux index and apply where the mask is true
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[:, :, self.flux_index] = True
        x_out = torch.where(mask, modified_flux, x)

        return x_out


class MultimodalSigmoid(nn.Module):
    def __init__(self, fitted_gaussians, flux_index):
        super(MultimodalSigmoid, self).__init__()
        self.flux_index = flux_index

        # Convert fitted Gaussian parameters to tensors
        self.amplitudes = torch.tensor([amp for amp, _, _ in fitted_gaussians], dtype=torch.float32)
        self.means = torch.tensor([mean for _, mean, _ in fitted_gaussians], dtype=torch.float32)
        self.stddevs = torch.tensor([stddev for _, _, stddev in fitted_gaussians], dtype=torch.float32)
        self.steepnesses = 1 / self.stddevs

    def gaussian_cdf(self, x, mean, std):
        return 0.5 * (1 + torch.erf((x.unsqueeze(-1) - mean) / (std * math.sqrt(2))))

    def compute_composite_sigmoid_cdf(self, x_flux, amplitudes, means, stddevs):
        # Compute CDF components
        cdf_components = self.gaussian_cdf(x_flux, means, stddevs)

        # Compute weighted sum of CDF components
        weighted_cdf_components = cdf_components * amplitudes
        composite_sigmoid = weighted_cdf_components.sum(dim=-1)

        # # Normalize the composite sigmoid
        amplitude_sum = amplitudes.sum()
        normalized_composite_sigmoid = composite_sigmoid / amplitude_sum

        return normalized_composite_sigmoid


    def compute_composite_sigmoid(self, x, amplitudes, means, steepnesses):

        exponents = -steepnesses * (x.unsqueeze(-1) - means)
        exponents = torch.clamp(exponents, min=-30, max=30)  # Clamp exponents to prevent overflow
        sigmoid_components = amplitudes / (1 + torch.exp(exponents))  # Compute sigmoid components
        composite_sigmoid = sigmoid_components.sum(dim=2)

        # Normalize the composite sigmoid by dividing it by the sum of the amplitudes
        amplitude_sum = amplitudes.sum()
        normalized_composite_sigmoid = composite_sigmoid / amplitude_sum

        return normalized_composite_sigmoid

    def forward(self, x):
        # x: Input tensor of shape [batch_size, height_size, num_fluxes]
        # We will apply the multimodal sigmoid to flux_index across all heights
        batch_size, height_size, num_fluxes = x.shape
        device = x.device
        dtype = x.dtype

        # Expand parameters for broadcasting
        amplitudes = self.amplitudes.view(1, 1, -1).to(device=device, dtype=dtype)    # Shape: [1, 1, n_params]
        means = self.means.view(1, 1, -1).to(device=device, dtype=dtype)              # Shape: [1, 1, n_params]
        steepnesses = self.steepnesses.view(1, 1, -1).to(device=device, dtype=dtype)  # Shape: [1, 1, n_params]
        stddevs = self.stddevs.view(1, 1, -1).to(device=device, dtype=dtype)  # Shape: [1, 1, n_params]

        # Extract the flux values for the specified flux index
        x_flux = x[:, :, self.flux_index]  # Shape: [batch_size, height_size]

        # Compute the composite sigmoid using the encapsulated method
        #composite_sigmoid = self.compute_composite_sigmoid_cdf(x_flux, amplitudes, means, stddevs)
        composite_sigmoid = self.compute_composite_sigmoid(x_flux, amplitudes, means, steepnesses)



         # Prepare a tensor to store the modified flux values
        modified_flux = composite_sigmoid  # Shape: [batch_size, height_size]
        modified_flux = modified_flux.unsqueeze(-1)  # Shape: [batch_size, height_size, 1]

        # Create a mask for the specified flux index
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[:, :, self.flux_index] = True
        x_out = torch.where(mask, modified_flux, x)

        return x_out