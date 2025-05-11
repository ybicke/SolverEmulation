"""
Diffusion model implementation for radiative flux prediction.

This module adapts the EDM (Elucidating Diffusion Model) framework to work
with the UNet architecture for atmospheric column data.
"""

import torch
import torch.nn as nn

from .base_methods import BaseRadiationModel
from .unet_diffusion import UNet


def append_dims(x, target_dims):
    """Appends dimensions to x until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}')
    return x[(...,) + (None,) * dims_to_append]


class EDM:
    """
    EDM (Elucidating Diffusion Model) implementation for radiative flux prediction.
    Based on the paper: "Elucidating the Design Space of Diffusion-Based Generative Models"
    """
    def __init__(self,
                 sigma_min=0.002,
                 sigma_max=80.0,
                 rho=7.0,
                 sigma_data=0.5,
                 P_mean=-1.2,
                 P_std=1.2,
                 S_churn=40,
                 S_min=0.05,
                 S_max=50,
                 S_noise=1.003):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_data = sigma_data
        self.P_mean = P_mean
        self.P_std = P_std
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise

    def sigma(self, eps):
        return (eps * self.P_std + self.P_mean).exp()

    def loss_weight(self, sigma):
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

    def skip_scaling(self, sigma):
        return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

    def out_scaling(self, sigma):
        return sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5

    def in_scaling(self, sigma):
        return 1 / (sigma**2 + self.sigma_data**2) ** 0.5

    def noise_conditioning(self, sigma):
        return 0.25 * sigma.log()

    def sampling_sigmas(self, num_steps, device=None):
        rho_inv = 1 / self.rho
        step_idxs = torch.arange(num_steps, device=device)
        sigmas = (
            self.sigma_max**rho_inv
            + step_idxs / (num_steps - 1) * (self.sigma_min**rho_inv - self.sigma_max**rho_inv)
        ) ** self.rho
        return torch.cat([sigmas, torch.zeros_like(sigmas[:1])])  # add sigma=0

    def sigma_hat(self, sigma, num_steps):
        gamma = (
            min(self.S_churn / num_steps, 2**0.5 - 1) if self.S_min <= sigma <= self.S_max else 0
        )
        return sigma + gamma * sigma


class RadiativeDiffusionModel(BaseRadiationModel):
    """
    Radiative flux diffusion model that combines the UNet architecture with EDM framework.
    """
    
    def __init__(
            self,
            height_in,
            channel_3d,
            channel_2d,
            channel_out,
            cnn_units,
            kernel_sizes,
            dropout,
            num_sampling_steps=25,
            deterministic_sampling=True,
            device='cuda',
            **kwargs
        ):
        """
        Initialize the diffusion model for radiative flux prediction.
        
        Args:
            height_in: Number of height levels in input
            channel_3d: Number of 3D input channels (features per height level)
            channel_2d: Number of 2D input channels (global features)
            channel_out: Number of output channels (default: 4 for [lw_up, lw_dn, sw_up, sw_dn])
            cnn_units: List of number of features in each CNN layer
            kernel_sizes: List of kernel sizes for max pooling at each level
            dropout: Dropout rate
            num_sampling_steps: Number of sampling steps during inference
            deterministic_sampling: Whether to use deterministic sampling 
            device: Device to run model on
            **kwargs: Additional arguments passed to BaseRadiationModel
        """
        super().__init__(**kwargs)
        
        self.height_in = height_in
        self.channel_out = channel_out
        self.channel_3d = channel_3d
        self.channel_2d = channel_2d
        self.device = device
        self.num_sampling_steps = num_sampling_steps
        self.deterministic_sampling = deterministic_sampling
        
        # Initialize the UNet backbone
        self.unet = UNet(
            height_in=height_in,
            channel_3d=channel_3d,
            channel_2d=channel_2d,
            channel_out=channel_out,
            cnn_units=cnn_units,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
            device=device
        )
        
        # Initialize the diffusion model
        self.edm = EDM()
        
        # Time embedding network (for noise level conditioning)
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
        )

    def forward(self, x3d_norm, x2d_norm, x2d_orig, sigma=None, noise=None):
        """
        Forward pass for training the diffusion model.
        
        Args:
            x3d_norm: Normalized 3D input data [batch_size, height, features_3d]
            x2d_norm: Normalized 2D input data [batch_size, features_2d]
            x2d_orig: Original 2D data for output scaling
            sigma: Noise level for diffusion (batch_size,)
            noise: Optional pre-generated noise
            
        Returns:
            Predicted denoised data
        """
        # Get ground truth from UNet (clean prediction without diffusion)
        with torch.no_grad():
            clean_pred = self.unet(x3d_norm, x2d_norm, x2d_orig)
        
        batch_size = x3d_norm.shape[0]
        
        # If no noise level provided, sample random noise levels
        if sigma is None:
            eps = torch.randn(batch_size, device=self.device)
            sigma = self.edm.sigma(eps)
        
        # Create noise if not provided
        if noise is None:
            noise = torch.randn_like(clean_pred) * append_dims(sigma, clean_pred.dim())
        
        # Add noise to the clean prediction
        noisy_sample = clean_pred + noise
        
        # Scale the input according to noise level
        sample_in = noisy_sample * append_dims(self.edm.in_scaling(sigma), noisy_sample.dim())
        
        # Get noise level conditioning
        t_emb = self.time_embed(self.edm.noise_conditioning(sigma).unsqueeze(1))
        
        # Get denoised prediction from UNet
        denoised = self.unet.forward_with_cond(x3d_norm, x2d_norm, x2d_orig, sample_in, t_emb)
        
        # Apply skip connection scaling
        skip = append_dims(self.edm.skip_scaling(sigma), noisy_sample.dim()) * noisy_sample
        
        # Apply output scaling
        scaled_output = denoised * append_dims(self.edm.out_scaling(sigma), denoised.dim()) + skip
        
        return scaled_output, clean_pred

    @torch.no_grad()
    def sample(self, x3d_norm, x2d_norm, x2d_orig):
        """
        Sample from the diffusion model using Heun's method.
        
        Args:
            x3d_norm: Normalized 3D input data
            x2d_norm: Normalized 2D input data
            x2d_orig: Original 2D data for output scaling
            
        Returns:
            Sample from the diffusion model
        """
        # Get shape from conditioning information
        batch_size = x3d_norm.shape[0]
        
        # Forward pass to get an estimate of the output shape
        with torch.no_grad():
            shape_est = self.unet(x3d_norm, x2d_norm, x2d_orig).shape
        
        # Sample initial noise
        sigmas = self.edm.sampling_sigmas(self.num_sampling_steps, device=self.device)
        
        # Start with pure noise
        eps = torch.randn(shape_est, device=self.device) * sigmas[0]
        
        # Choose sampling method
        if self.deterministic_sampling:
            sample = self.sample_deterministically(eps, sigmas, x3d_norm, x2d_norm, x2d_orig)
        else:
            sample = self.sample_stochastically(eps, sigmas, x3d_norm, x2d_norm, x2d_orig)
            
        return sample

    def sample_deterministically(self, eps, sigmas, x3d_norm, x2d_norm, x2d_orig):
        """
        Deterministic sampling using Heun's 2nd order method.
        """
        sample_next = eps
        
        for i, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
            sample_curr = sample_next
            
            # Condition on noise level
            t_emb = self.time_embed(self.edm.noise_conditioning(sigma).repeat(sample_curr.shape[0]).unsqueeze(1))
            
            # Scale input
            sample_in = sample_curr * append_dims(self.edm.in_scaling(sigma), sample_curr.dim())
            
            # Get prediction from UNet
            denoised = self.unet.forward_with_cond(x3d_norm, x2d_norm, x2d_orig, sample_in, t_emb)
            
            # Apply skip and output scaling
            pred_curr = denoised * append_dims(self.edm.out_scaling(sigma), denoised.dim())
            pred_curr = pred_curr + append_dims(self.edm.skip_scaling(sigma), sample_curr.dim()) * sample_curr
            
            # Euler step
            d_cur = (sample_curr - pred_curr) / sigma
            sample_next = sample_curr + d_cur * (sigma_next - sigma)
            
            # Second order correction (Heun's method)
            if i < self.num_sampling_steps - 1:
                # Condition on next noise level
                t_emb_next = self.time_embed(self.edm.noise_conditioning(sigma_next).repeat(sample_next.shape[0]).unsqueeze(1))
                
                # Scale input
                sample_in_next = sample_next * append_dims(self.edm.in_scaling(sigma_next), sample_next.dim())
                
                # Get prediction from UNet
                denoised_next = self.unet.forward_with_cond(x3d_norm, x2d_norm, x2d_orig, sample_in_next, t_emb_next)
                
                # Apply skip and output scaling
                pred_next = denoised_next * append_dims(self.edm.out_scaling(sigma_next), denoised_next.dim())
                pred_next = pred_next + append_dims(self.edm.skip_scaling(sigma_next), sample_next.dim()) * sample_next
                
                # Apply correction
                d_prime = (sample_next - pred_next) / sigma_next
                sample_next = sample_curr + (sigma_next - sigma) * (0.5 * d_cur + 0.5 * d_prime)
        
        return sample_next

    def sample_stochastically(self, eps, sigmas, x3d_norm, x2d_norm, x2d_orig):
        """
        Stochastic sampling with noise perturbation at each step.
        """
        sample_next = eps
        
        for i, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
            sample_curr = sample_next
            
            # Increase noise temporarily
            sigma_hat = self.edm.sigma_hat(sigma, self.num_sampling_steps)
            noise = torch.randn_like(sample_curr) * self.edm.S_noise
            sample_hat = sample_curr + noise * (sigma_hat**2 - sigma**2) ** 0.5
            
            # Condition on noise level
            t_emb = self.time_embed(self.edm.noise_conditioning(sigma_hat).repeat(sample_hat.shape[0]).unsqueeze(1))
            
            # Scale input
            sample_in = sample_hat * append_dims(self.edm.in_scaling(sigma_hat), sample_hat.dim())
            
            # Get prediction from UNet
            denoised = self.unet.forward_with_cond(x3d_norm, x2d_norm, x2d_orig, sample_in, t_emb)
            
            # Apply skip and output scaling
            pred_hat = denoised * append_dims(self.edm.out_scaling(sigma_hat), denoised.dim())
            pred_hat = pred_hat + append_dims(self.edm.skip_scaling(sigma_hat), sample_hat.dim()) * sample_hat
            
            # Euler step
            d_cur = (sample_hat - pred_hat) / sigma_hat
            sample_next = sample_hat + d_cur * (sigma_next - sigma_hat)
            
            # Second order correction
            if i < self.num_sampling_steps - 1:
                # Condition on next noise level
                t_emb_next = self.time_embed(self.edm.noise_conditioning(sigma_next).repeat(sample_hat.shape[0]).unsqueeze(1))
                
                # Scale input
                sample_in_next = sample_next * append_dims(self.edm.in_scaling(sigma_next), sample_next.dim())
                
                # Get prediction from UNet
                denoised_next = self.unet.forward_with_cond(x3d_norm, x2d_norm, x2d_orig, sample_in_next, t_emb_next)
                
                # Apply skip and output scaling
                pred_next = denoised_next * append_dims(self.edm.out_scaling(sigma_next), denoised_next.dim())
                pred_next = pred_next + append_dims(self.edm.skip_scaling(sigma_next), sample_next.dim()) * sample_next
                
                # Apply correction
                d_prime = (sample_next - pred_next) / sigma_next
                sample_next = sample_hat + (sigma_next - sigma_hat) * (0.5 * d_cur + 0.5 * d_prime)
        
        return sample_next

    def training_step(self, x3d_norm, x2d_norm, x2d_orig):
        """
        Single training step for the diffusion model.
        
        Args:
            x3d_norm: Normalized 3D input data
            x2d_norm: Normalized 2D input data
            x2d_orig: Original 2D data for output scaling
            
        Returns:
            Loss value
        """
        batch_size = x3d_norm.shape[0]
        
        # Sample random noise levels
        eps = torch.randn(batch_size, device=self.device)
        sigma = self.edm.sigma(eps)
        
        # Run forward pass
        denoised, target = self.forward(x3d_norm, x2d_norm, x2d_orig, sigma)
        
        # Calculate mean squared error loss
        loss = (denoised - target) ** 2
        
        # Apply loss weighting
        weight = append_dims(self.edm.loss_weight(sigma), loss.dim())
        
        return (loss * weight).mean()