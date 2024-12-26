import torch
import torch.nn as nn
import numpy as np


import torch
from torch.utils.data import DataLoader
from data_loaders_new import IconColumnIterableDataset
import matplotlib.pyplot as plt
import os
import numpy as np
from os.path import join
import glob
import re
import random


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
        self.adjusted_means = []
        self.adjusted_steepnesses = []
        self.overall_means = []
        self.overall_stddevs = []

        for h in self.heights:
            params = gaussian_params_by_height[h]
            amplitudes = torch.tensor([amp for amp, _, _ in params], dtype=torch.float32)
            means = torch.tensor([mean for _, mean, _ in params], dtype=torch.float32)
            stddevs = torch.tensor([stddev for _, _, stddev in params], dtype=torch.float32)
            epsilon = 1e-8
            steepnesses = 3 / (stddevs + epsilon) # Calculate steepnesses based on standard deviations

             # Compute overall mean and standard deviation for the current height level
            total_amplitude = amplitudes.sum()
            overall_mean = (amplitudes * means).sum() / total_amplitude
            variance = (amplitudes * ((means - overall_mean) ** 2 + stddevs ** 2)).sum() / total_amplitude
            overall_stddev = torch.sqrt(variance + epsilon)

            # Adjust means and steepnesses for centering and scaling
            adjusted_means = (means - overall_mean) / overall_stddev
            adjusted_steepnesses = steepnesses * overall_stddev

            # Store parameters for each height
            self.amplitudes.append(amplitudes)
            self.adjusted_means.append(adjusted_means)
            self.adjusted_steepnesses.append(adjusted_steepnesses)
            self.overall_means.append(overall_mean)
            self.overall_stddevs.append(overall_stddev)


    def compute_composite_sigmoid(self, x, amplitudes, means, steepnesses):
        exponents = -steepnesses * (x.unsqueeze(1) - means)
        exponents = torch.clamp(exponents, min=-30, max=30)  # Clamp exponents to prevent overflow
        sigmoid_components = amplitudes / (1 + torch.exp(exponents))  # Compute sigmoid components
        composite_sigmoid = sigmoid_components.sum(dim=1)

        # Normalize the composite sigmoid by dividing it by the sum of the amplitudes
        amplitude_sum = amplitudes.sum()
        normalized_composite_sigmoid = composite_sigmoid / amplitude_sum

        return normalized_composite_sigmoid


    def forward(self, x):
        batch_size, height_size, num_fluxes = x.shape
        device = x.device
        dtype = x.dtype

        modified_flux = torch.zeros(batch_size, height_size, device=device, dtype=dtype)

        for idx in range(height_size):  # idx from 0 to height_size - 1
            phys_idx = idx + 1  # Adjust index if needed

            if phys_idx in self.heights:
                # Get parameters for specific height index
                height_idx = self.heights.index(phys_idx)
                amplitudes = self.amplitudes[height_idx].to(device=device, dtype=dtype)
                adjusted_means = self.adjusted_means[height_idx].to(device=device, dtype=dtype)
                adjusted_steepnesses = self.adjusted_steepnesses[height_idx].to(device=device, dtype=dtype)
                overall_mean = self.overall_means[height_idx].to(device=device, dtype=dtype)
                overall_stddev = self.overall_stddevs[height_idx].to(device=device, dtype=dtype)

                # Extract the flux values at this height
                x_h = x[:, idx, self.flux_index]

                # Shift and scale x_h
                adjusted_x_h = (x_h - overall_mean) / overall_stddev

                # Compute the composite sigmoid using the adjusted parameters
                composite_sigmoid = self.compute_composite_sigmoid(
                    adjusted_x_h, amplitudes, adjusted_means, adjusted_steepnesses
                )
                modified_flux[:, idx] = composite_sigmoid
            else:
                # Apply standard sigmoid for the remaining height layers
                x_h = x[:, idx, self.flux_index]
                modified_flux[:, idx] = torch.sigmoid(x_h)

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
        self.setup_parameters(fitted_gaussians)

    def setup_parameters(self, fitted_gaussians):
        # Convert fitted Gaussian parameters to tensors
        amplitudes = torch.tensor([amp for amp, _, _ in fitted_gaussians], dtype=torch.float32)
        means = torch.tensor([mean for _, mean, _ in fitted_gaussians], dtype=torch.float32)
        stddevs = torch.tensor([stddev for _, _, stddev in fitted_gaussians], dtype=torch.float32)
        epsilon = 1e-8
        steepnesses = 3 / (stddevs + epsilon)

        # Compute overall mean and standard deviation
        total_amplitude = amplitudes.sum()
        overall_mean = (amplitudes * means).sum() / total_amplitude
        variance = (amplitudes * ((means - overall_mean) ** 2 + stddevs ** 2)).sum() / total_amplitude
        overall_stddev = torch.sqrt(variance + epsilon)

        # Adjust means and steepnesses for centering and scaling
        adjusted_means = (means - overall_mean) / overall_stddev
        adjusted_steepnesses = steepnesses * overall_stddev

        # Store parameters as buffers
        self.register_buffer('amplitudes', amplitudes)
        self.register_buffer('adjusted_means', adjusted_means)
        self.register_buffer('adjusted_steepnesses', adjusted_steepnesses)
        self.register_buffer('overall_mean', torch.tensor(overall_mean))
        self.register_buffer('overall_stddev', torch.tensor(overall_stddev))


    def compute_composite_sigmoid(self, x_flux, amplitudes, means, steepnesses):
        # Reshape for broadcasting
        x_flux_expanded = x_flux.unsqueeze(-1)  # Shape: [batch_size, height_size, 1]
        exponents = -steepnesses * (x_flux_expanded - means)
        exponents = torch.clamp(exponents, min=-50, max=50)
        sigmoid_components = amplitudes / (1 + torch.exp(exponents))  # Shape: [batch_size, height_size, n_params]
        composite_sigmoid = sigmoid_components.sum(dim=-1)  # Shape: [batch_size, height_size]

        # Normalize the composite sigmoid by dividing it by the sum of the amplitudes
        amplitude_sum = amplitudes.sum()
        normalized_composite_sigmoid = composite_sigmoid / amplitude_sum

        return normalized_composite_sigmoid

    def forward(self, x):
        batch_size, height_size, num_fluxes = x.shape
        device = x.device
        dtype = x.dtype

        # Extract the flux values for the specified flux index
        x_flux = x[:, :, self.flux_index]  # Shape: [batch_size, height_size]

        # Shift and scale x_flux
        adjusted_x_flux = (x_flux - self.overall_mean) / self.overall_stddev  # Shape: [batch_size, height_size]

        # Compute the composite sigmoid using the adjusted x_flux and stored parameters
        composite_sigmoid = self.compute_composite_sigmoid(
            adjusted_x_flux,
            self.amplitudes.to(device=device, dtype=dtype),
            self.adjusted_means.to(device=device, dtype=dtype),
            self.adjusted_steepnesses.to(device=device, dtype=dtype)
        )

        # Prepare a tensor to store the modified flux values
        modified_flux = composite_sigmoid.unsqueeze(-1)  # Shape: [batch_size, height_size, 1]

        # Create a mask for the specified flux index and apply the modifications
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[:, :, self.flux_index] = True
        x_out = torch.where(mask, modified_flux, x)

        return x_out