# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import logging



class AFNO1D(nn.Module):
    """
    hidden_size: channel dimension size, i.e. feature dimension each data point will have after preprocessing 
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    """
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.05, hard_thresholding_fraction=1, hidden_size_factor=1, is_test=False):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.is_test = is_test

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks # 256 / 8 = 32
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02
        
        

        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        
    
    def gaussian_low_pass_filter(self, x, N):
        B, num_modes, num_blocks, block_size, _ = x.shape
        frequencies = torch.fft.rfftfreq(N, d=1.0)[:num_modes]
        nyquist_frequency = 0.5 * N
        cutoff = 0.005 * nyquist_frequency
        sigma = cutoff / 3  # Standard deviation of the Gaussian function
        gaussian_filter = torch.exp(-0.5 * (frequencies / sigma) ** 2)
        gaussian_filter = gaussian_filter.view(1, -1, 1, 1, 1).to(x.device)
        return x * gaussian_filter
            
    logging.basicConfig(level=logging.INFO)
    


    def visualize_features(self, x_before_fft, x_after_fft, x_after_ifft):
        plt.figure(figsize=(12, 8))

        # Detach tensors and convert to numpy
        x_before_fft = x_before_fft.detach().cpu().numpy()
        x_after_fft = x_after_fft.detach().cpu().numpy()
        x_after_ifft = x_after_ifft.detach().cpu().numpy()

        # Normalize the data before plotting
        epsilon = 1e-8  # Small value to prevent division by zero
        x_before_fft_norm = (x_before_fft - x_before_fft.min()) / (x_before_fft.max() - x_before_fft.min() + epsilon)
        x_after_fft_norm = (np.abs(x_after_fft) - np.abs(x_after_fft).min()) / (np.abs(x_after_fft).max() - np.abs(x_after_fft).min() + epsilon)
        x_after_ifft_norm = (x_after_ifft - x_after_ifft.min()) / (x_after_ifft.max() - x_after_ifft.min() + epsilon)

        # Plot time-domain data for all features
        plt.subplot(2, 2, 1)
        plt.imshow(x_before_fft_norm, aspect='auto', cmap='viridis')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('Time Domain All Features')
        plt.xlabel('Feature Index')
        plt.ylabel('Height Level')
        plt.clim(0, 1)

        # Plot magnitude of frequency-domain data for all features
        plt.subplot(2, 2, 2)
        plt.imshow(x_after_fft_norm, aspect='auto', cmap='viridis')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('Magnitude Frequency All Features')
        plt.xlabel('Feature Index')
        plt.ylabel('Frequency Component')
        plt.clim(0, 1)

        # Plot time-domain data after inverse FFT
        plt.subplot(2, 2, 3)
        plt.imshow(x_after_ifft_norm, aspect='auto', cmap='viridis')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('Time Domain After Inverse FFT')
        plt.xlabel('Feature Index')
        plt.ylabel('Height Level')
        plt.clim(0, 1)

        plt.tight_layout()
        
        
        model_name = 'afno_column_1percent_8_concat_easy'
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'{model_name}.png'
        
        
        plot_dir = '/mydata/deepcloud/yves/online-datasets/workspace/results/plots/2d_embedding'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        plt.savefig(os.path.join(plot_dir, filename))
        plt.close()

        
    
    def visualize_features_signals(self, x_before_fft, x_after_fft, x_after_ifft, x_before_sparsification, x_after_sparsification):
        num_features = x_before_fft.shape[1]
        num_rows = 5  # Reduced from 6 to 5
        num_cols = num_features

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*4, num_rows*3))

        # Detach tensors and convert to numpy
        x_before_fft = x_before_fft.detach().cpu().numpy()
        x_after_fft = x_after_fft.detach().cpu().numpy()
        x_after_ifft = x_after_ifft.detach().cpu().numpy()
        x_before_sparsification = x_before_sparsification.detach().cpu().numpy()
        x_after_sparsification = x_after_sparsification.detach().cpu().numpy()

        for feature_idx in range(num_features):
            # Plot time-domain data for each feature
            axes[0, feature_idx].plot(x_before_fft[:, feature_idx])
            axes[0, feature_idx].set_title(f'Time Domain Feature {feature_idx}')
            axes[0, feature_idx].set_xlabel('Height Level')
            axes[0, feature_idx].set_ylabel('Value')

            # Plot magnitude of frequency-domain data for each feature
            axes[1, feature_idx].plot(np.abs(x_after_fft[:, feature_idx]))
            axes[1, feature_idx].set_title(f'Magnitude Frequency Feature {feature_idx}')
            axes[1, feature_idx].set_xlabel('Frequency Component')
            axes[1, feature_idx].set_ylabel('Magnitude')

            # Plot magnitude of frequency-domain data before sparsification for each feature
            axes[2, feature_idx].plot(np.abs(x_before_sparsification[:, feature_idx].squeeze()))
            axes[2, feature_idx].set_title(f'Magnitude Freq Before Sparsification {feature_idx}')
            axes[2, feature_idx].set_xlabel('Frequency Component')
            axes[2, feature_idx].set_ylabel('Magnitude')

            # Plot magnitude of frequency-domain data after sparsification for each feature
            axes[3, feature_idx].plot(np.abs(x_after_sparsification[:, feature_idx].squeeze()))
            axes[3, feature_idx].set_title(f'Magnitude Freq After Sparsification {feature_idx}')
            axes[3, feature_idx].set_xlabel('Frequency Component')
            axes[3, feature_idx].set_ylabel('Magnitude')

            # Plot time-domain data after inverse FFT for each feature
            axes[4, feature_idx].plot(x_after_ifft[:, feature_idx])
            axes[4, feature_idx].set_title(f'After Inverse FFT Feature {feature_idx}')
            axes[4, feature_idx].set_xlabel('Height Level')
            axes[4, feature_idx].set_ylabel('Value')

        plt.tight_layout()


        model_name = 'afno_column_1percent_8_concat_easy'
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'{model_name}.png'

        plot_dir = '/mydata/deepcloud/yves/online-datasets/workspace/results/plots/signals_trained'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        plt.savefig(os.path.join(plot_dir, filename))
        plt.close()
    
    
    def forward(self, x):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, N, C = x.shape 

        # F(X)
        x_before_fft = x.clone()
        x = torch.fft.rfft(x, dim=1, norm="ortho") 
        x_after_fft = x.clone() 
        
        # real fft  returns only the non-negative frequency components due to symmetry. Real FFT along the second dimension of x, orhtho returns the fourier coefficent normalized st. FT is unitary.
        # Preservs L2 norm of original data. When set to "ortho", the function returns the Fourier coefficients normalized such that the Fourier transform is unitary (preservs the length of vectors).
        # The energy measured by the L2 norm is the same in time and frequency domain. This normalization ensures that the overall energy of the signal is not altered

        x = x.reshape(B, N // 2 + 1, self.num_blocks, self.block_size) # preparation for block-wise processing 

        o1_real = torch.zeros([B, N // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, N // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)
        # A larger hidden_size_factor would increase the number of features per block, potentially allowing the model to capture 
        # more detailed information but at the cost of increased computational requirements.

        total_modes = N // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        # F(k)*F(X) with non linearity
        # [b, i] [i, o] --> [b, o] , sum product over i between two tensors, i up to kept modes
        o1_real[:, :kept_modes] = F.relu( 
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )
        
        o1_imag[:, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        # x[:, :kept_modes].real and x[:, :kept_modes].imag represent the real and imaginary parts of these Fourier
        # coefficients, respectively, for a subset of modes (frequencies) that are kept after the transform.

        # Using multiple transformations in the FNO framework, such as o1 followed by o2, enhances the model's expressiveness
        # by increasing its depth and parameter space. This allows for capturing more complex patterns and dependencies within the data

        o2_real[:, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)

        # low pass filtering
        #x_before_low_pass = x.clone() 
        #x = self.gaussian_low_pass_filter(x, N)
        #x_after_low_pass = x.clone() 
        
        # Sparsity application
        x_before_sparsification = x.clone()
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x_after_sparsification = x.clone() 
        
        x = torch.view_as_complex(x)
        x = x.reshape(B, N // 2 + 1, C)
        x_after_mixing = x.clone() 

        # look at frequency domain for 4 dimensions befor ifft
        #x_real = x.real
        #x_imag = x.imag
        #x_real = self.mlp_head(x_real)
        #x_imag = self.mlp_head(x_imag)

        # Recombine the real and imaginary parts
        # x = torch.complex(x_real, x_imag)
        # x_after_reshaping = x.clone()

        # F^-1(F(k)*F(X))(s) = K(X)(s)
        x = torch.fft.irfft(x, n=N, dim=1, norm="ortho")
        x_after_ifft = x.clone() 

        # x = torch.fft.ifft(x, dim=1, norm="ortho").real  # Convert back to real by taking the real part
            # Ensure bias has the same shape as x
        # bias = bias[:, :x.shape[1], :x.shape[2]]
        
        x = x.type(dtype)
        output = x + bias

        # for visualization directly in the mixer
        self.visualize_features(x_before_fft[0], x_after_fft[0], x_after_ifft[0])
        self.visualize_features_signals(x_before_fft[0], x_after_fft[0], x_after_ifft[0], x_before_sparsification[0], x_after_sparsification[0])

        return output
       