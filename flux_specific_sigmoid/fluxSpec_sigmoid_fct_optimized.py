import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
import random

# class MultimodalSigmoid(nn.Module):
#     def __init__(self, fitted_gaussians, flux_index):
#         super(MultimodalSigmoid, self).__init__()
#         self.flux_index = flux_index  # Index of the flux to apply the sigmoid to

#         # Convert fitted Gaussian parameters to tensors
#         amplitudes = torch.tensor([amp for amp, _, _ in fitted_gaussians], dtype=torch.float32)
#         means = torch.tensor([mean for _, mean, _ in fitted_gaussians], dtype=torch.float32)
#         stddevs = torch.tensor([stddev for _, _, stddev in fitted_gaussians], dtype=torch.float32)
#         steepnesses = 3 / stddevs

#         # Store parameters as buffers (non-trainable tensors)
#         self.register_buffer('amplitudes', amplitudes)
#         self.register_buffer('means', means)
#         self.register_buffer('steepnesses', steepnesses)

#     # TODO: Could be optimized more
#     def forward(self, x):
#         # x: Input tensor of shape [batch_size, height_size, num_fluxes]
#         # We will apply the multimodal sigmoid to flux_index across all heights
        
#         # Create a copy of x to avoid in-place modification
#         x_out = x.clone()

#         # Extract the flux values for the specified flux index
#         x_flux = x_out[:, :, self.flux_index]  # Shape: [batch_size, height_size]

#         # Reshape for broadcasting
#         x_flux_expanded = x_flux.unsqueeze(-1)  # Shape: [batch_size, height_size, 1]

#         # Expand parameters for broadcasting
#         amplitudes = self.amplitudes.view(1, 1, -1)    # Shape: [1, 1, n_params]
#         means = self.means.view(1, 1, -1)              # Shape: [1, 1, n_params]
#         steepnesses = self.steepnesses.view(1, 1, -1)  # Shape: [1, 1, n_params]

#         # Compute exponents for the sigmoid function
#         exponents = -steepnesses * (x_flux_expanded - means)  # Shape: [batch_size, height_size, n_params]
#         exponents = torch.clamp(exponents, min=-100, max=100)

#         # Compute sigmoid components and sum across the parameters
#         sigmoid_components = amplitudes / (1 + torch.exp(exponents))  # Shape: [batch_size, height_size, n_params]
#         composite_sigmoid = sigmoid_components.sum(dim=-1)            # Shape: [batch_size, height_size]

#         # Normalize the composite sigmoid
#         max_value = composite_sigmoid.max()
#         max_value = max_value if max_value != 0 else 1  # Avoid division by zero
#         composite_sigmoid = composite_sigmoid / max_value 
        
#         # Update the output tensor
#         x_out[:, :, self.flux_index] = composite_sigmoid

#         return x_out    


class MultimodalSigmoid(nn.Module):
    def __init__(self, fitted_gaussians, flux_index):
        super(MultimodalSigmoid, self).__init__()
        self.flux_index = flux_index

        # Convert fitted Gaussian parameters to tensors
        self.amplitudes = nn.Parameter(torch.tensor([amp for amp, _, _ in fitted_gaussians], dtype=torch.float32), requires_grad=False)
        self.means = nn.Parameter(torch.tensor([mean for _, mean, _ in fitted_gaussians], dtype=torch.float32), requires_grad=False)
        stddevs = torch.tensor([stddev for _, _, stddev in fitted_gaussians], dtype=torch.float32)
        self.steepnesses = nn.Parameter(3 / stddevs, requires_grad=False)

    def compute_composite_sigmoid(self, x_flux, amplitudes, means, steepnesses):
        # Reshape for broadcasting
        x_flux_expanded = x_flux.unsqueeze(-1)  # Shape: [batch_size, height_size, 1]
        exponents = -steepnesses * (x_flux_expanded - means)
        exponents = torch.clamp(exponents, min=-100, max=100)
        sigmoid_components = amplitudes / (1 + torch.exp(exponents))  # Shape: [batch_size, height_size, n_params]
        composite_sigmoid = sigmoid_components.sum(dim=-1)            # Shape: [batch_size, height_size]
        max_value = composite_sigmoid.max()
        max_value = max_value if max_value != 0 else 1e-8
        composite_sigmoid = composite_sigmoid / max_value
        
        return composite_sigmoid

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

        # Extract the flux values for the specified flux index
        x_flux = x[:, :, self.flux_index]  # Shape: [batch_size, height_size]

        # Compute the composite sigmoid using the encapsulated method
        composite_sigmoid = self.compute_composite_sigmoid(x_flux, amplitudes, means, steepnesses)

         # Prepare a tensor to store the modified flux values
        modified_flux = composite_sigmoid  # Shape: [batch_size, height_size]
        modified_flux = modified_flux.unsqueeze(-1)  # Shape: [batch_size, height_size, 1]

        # Create a mask for the specified flux index
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[:, :, self.flux_index] = True
        x_out = torch.where(mask, modified_flux, x)

        return x_out
    
    

def load_gaussian_parameters(file_path):
    data = np.load(file_path)
    amplitudes = data['amplitudes']
    means = data['means']
    stddevs = data['stddevs']
    fitted_gaussians = list(zip(amplitudes, means, stddevs))
    print(f"Fitted Gaussian parameters loaded from {file_path}")
    return fitted_gaussians




def plot_multimodal_sigmoid(multimodal_sigmoid, flux_index, save_path):
    """
    Plot the multimodal sigmoid function applied to a range of flux values at the specified flux index.
    
    Args:
        multimodal_sigmoid (MultimodalSigmoid): The multimodal sigmoid module.
        flux_index (int): The index of the flux to which the sigmoid is applied.
        save_path (str): The path where the plot will be saved.
    """
    # Define the range of input flux values
    x_values = np.linspace(0, 400, 1000)  # Adjust the range based on your data
    num_points = len(x_values)
    batch_size = num_points
    num_fluxes = 4  # Adjust based on your actual number of fluxes
    height_size = 1  # Since the function is not height-dependent, use height_size = 1

    # Create an input tensor with zeros
    x_input = torch.zeros(batch_size, height_size, num_fluxes, dtype=torch.float32)

    # Set the flux index to x_values
    x_input[:, 0, flux_index] = torch.tensor(x_values, dtype=torch.float32)

    # Apply the multimodal sigmoid function
    with torch.no_grad():
        y_output = multimodal_sigmoid(x_input)

    # Extract the output for the flux index
    y_values = y_output[:, 0, flux_index].numpy()

    # Plot the sigmoid function
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, label='Multimodal Sigmoid', color='blue', linewidth=2)


    plt.title('Multimodal Sigmoid Function')
    plt.xlabel('Flux Value')
    plt.ylabel('Sigmoid Output')
    plt.grid(True)
    plt.legend()

    # Save the plot
    plt.savefig(save_path)
    plt.close()
    print(f"Multimodal sigmoid plot saved to {save_path}")

def main():
    # Set the random seed and RNG states
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load the Gaussian parameters for LW Up flux
    fitted_gaussians_file_lwup = '/mydata/deepcloud/yves/results_git/data_histograms/histogram_fit_up/fitted_gaussians_lwup_single_layer.npz'
    fitted_gaussians_lwup = load_gaussian_parameters(fitted_gaussians_file_lwup)

    # Initialize the MultimodalSigmoid module for LW Up flux (flux_index=0)
    flux_index = 0  # LW Up flux index
    multimodal_sigmoid = MultimodalSigmoid(fitted_gaussians_lwup, flux_index=flux_index)

    # Example: apply the multimodal sigmoid to some data
    # Replace this with your actual data loading and processing

    # Simulate some input data
    batch_size = 10
    height_size = 71
    num_fluxes = 4  # Adjust based on your actual number of fluxes

    # Create dummy input data
    x = torch.randn(batch_size, height_size, num_fluxes)

    # Apply the multimodal sigmoid function
    y = multimodal_sigmoid(x)
    
    # Plot the multimodal sigmoid function
    plot_save_path = os.path.splitext(fitted_gaussians_file_lwup)[0] + '_sigmoid_plot.png'
    plot_multimodal_sigmoid(multimodal_sigmoid, flux_index=flux_index, save_path=plot_save_path)

if __name__ == '__main__':
    main()