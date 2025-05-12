"""
U-Net CNN model for radiation flux prediction with diffusion model conditioning

This module implements a 1D CNN U-Net model specifically designed for diffusion models,
with time embedding for noise level conditioning during the denoising process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_methods import BaseRadiationModel


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, kernel_size=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff = x2.size()[2] - x1.size()[2]
        # Padding handling for when dimensions don't match
        x1 = F.pad(x1, [diff//2, diff-diff//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Maps feature space to output space"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class TimeEmbedding(nn.Module):
    """
    Time embedding for diffusion models
    
    Allows the network to understand what noise level it's working 
    with during the denoising process.
    """
    def __init__(self, dim):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, t):
        # t is expected to be a 1D tensor of shape [batch_size]
        # or a 2D tensor of shape [batch_size, 1]
        if t.dim() == 1:
            t = t.unsqueeze(1)
        return self.time_embed(t)


class UNetDiffusion(BaseRadiationModel):
    """
    U-Net architecture specifically designed for diffusion models with time conditioning
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
            device,
            time_embed_dim,
            **kwargs
        ):
        """
        Initialize the diffusion U-Net model for radiative flux prediction.
        
        Args:
            height_in: Number of height levels in input
            channel_3d: Number of 3D input channels (features per height level)
            channel_2d: Number of 2D input channels (global features)
            channel_out: Number of output channels (default: 4 for [lw_up, lw_dn, sw_up, sw_dn])
            cnn_units: List of number of features in each CNN layer
            kernel_sizes: List of kernel sizes for max pooling at each level
            dropout: Dropout rate
            device: Device to run model on
            time_embed_dim: Dimension for time embedding
            **kwargs: Additional arguments passed to BaseRadiationModel
        """
        super().__init__(**kwargs)
        
        self.height_in = height_in
        self.channel_out = channel_out
        self.channel_3d = channel_3d
        self.channel_2d = channel_2d
        self.device = device
        
        channel_in = channel_3d + channel_2d + channel_out
        
        # Time embedding components - critical for diffusion model
        self.time_embed_dim = time_embed_dim
        self.time_embedding = TimeEmbedding(time_embed_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, cnn_units[-1]),
            nn.SiLU()
        )
        
        self.inc = DoubleConv(channel_in, cnn_units[0])
        
        self.downs = nn.ModuleList()
        for i in range(0, len(cnn_units)-1):      
            self.downs.append(Down(cnn_units[i], cnn_units[i+1], kernel_sizes[i]))
        
        self.ups = nn.ModuleList()
        for i in range(len(cnn_units)-1, 0, -1):
            self.ups.append(Up(cnn_units[i], cnn_units[i-1]))
        
        self.outc = OutConv(cnn_units[0], channel_out)
        self.last_height = OutConv(height_in, height_in)
        self.sigmoid = nn.Sigmoid()
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, noisy_sample, time_cond, cond):
        """
        Diffusion model forward pass
        
        Args:
            noisy_sample: Noisy data sample that needs denoising
            time_cond: Time conditioning for noise level
            cond: Dictionary with conditioning information:
                  - "x3d_norm": Normalized 3D atmospheric features
                  - "x2d_norm": Normalized 2D global features
                  - "x2d_orig": Original 2D features (for output scaling)
            
        Returns:
            Denoised prediction
        """
        x3d_norm = cond["x3d_norm"]
        x2d_norm = cond["x2d_norm"]
        x2d_orig = cond["x2d_orig"]
        
        # Process conditioning inputs
        x2d = x2d_norm.unsqueeze(1).repeat(1, x3d_norm.shape[1], 1)
        x3d_norm = torch.cat([x2d, x3d_norm], dim=-1)
        ones = torch.ones(x3d_norm.shape[0], 1, x3d_norm.shape[-1], device=x3d_norm.device)
        x_cond = torch.cat([ones, x3d_norm], dim=1)
        
        if self.dropout:
            x_cond = self.dropout(x_cond)
        
        # Convert to channels-first format
        x_cond = x_cond.permute(0, 2, 1)  # [B, C, H]
        
        # Ensure noisy_sample has the right format
        if noisy_sample.shape[1] != self.channel_out or noisy_sample.shape[2] != self.height_in:
            # Permute if dimensions don't match expected format
            if noisy_sample.shape[2] == self.channel_out and noisy_sample.shape[1] == self.height_in:
                noisy_sample = noisy_sample.permute(0, 2, 1)
        
        # IMPORTANT: Concatenate the noisy sample with conditioning
        # This requires adjusting the input channels in your DoubleConv
        x = torch.cat([noisy_sample, x_cond], dim=1)
        
        # Process through U-Net
        skip = []
        features = self.inc(x)
        skip.append(features)
        
        # Encoder path
        for down in self.downs:
            skip.append(down(skip[-1]))
        
        # Bottleneck with time conditioning
        y = skip[-1]
        
        # Process time embedding
        t_emb = self.time_embedding(time_cond)
        t_emb = self.time_proj(t_emb)
        
        # Inject time information into bottleneck features
        t_emb = t_emb.unsqueeze(-1).repeat(1, 1, y.shape[-1])
        y = y + t_emb
        
        # Decoder path
        skip.pop()
        for up in self.ups:
            y = up(y, skip.pop())
        
        # Output processing
        y = self.outc(y).permute(0, 2, 1)
        y = self.last_height(y)
        y = self.sigmoid(y)
        
        # Scale output using physical quantities
        y = self._scale_output(y, x2d_orig)
            
        return y.squeeze()