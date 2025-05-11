"""
U-Net CNN model for radiation flux prediction
Adapted for diffusion model conditioning

This module implements a 1D CNN U-Net model for predicting radiative fluxes
in atmospheric columns, with support for time/noise conditioning for diffusion models.
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
        # maxpooling is done over the height dimension and the spatial dimension gets reduced
        # while spatial resolution decreases, the number of feature channels typically increases
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
# As spatial resolution decreases, you increase the feature channels to maintain model capacity
# Deeper layers with more channels can capture more complex patterns across a wider receptive field
# The skip connections then help recover the spatial details during upsampling


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
    # Maps feature space to output space
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# TODO: Check if this is correct
# Time embedding allows the network to understand what noise level (or diffusion timestep) 
# it's working with during the denoising process.
# It's a way to inject information about the diffusion process into the model.
class TimeEmbedding(nn.Module):
    """Time embedding for diffusion models"""
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


class UNet(BaseRadiationModel):
    
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
        Initialize the CNN model for radiative flux prediction.
        
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
        
        # Total input channels include 3D features and 2D features (broadcast to each height level)
        channel_in = channel_3d + channel_2d
        
        # TODO: Check if this is correct
        # Add time embedding for diffusion model
        self.time_embed_dim = time_embed_dim
        self.time_embedding = TimeEmbedding(time_embed_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, cnn_units[-1]),
            nn.SiLU()
        )
        
        # Initial double convolution
        self.inc = DoubleConv(channel_in, cnn_units[0])
        
        # Downsampling path
        # The downsampling path reduces spatial resolution while increasing feature channels
        # This allows the model to capture increasingly complex patterns as it goes deeper
        self.downs = nn.ModuleList()
        for i in range(0, len(cnn_units)-1):      
            self.downs.append(Down(cnn_units[i], cnn_units[i+1], kernel_sizes[i]))
        
        # Upsampling path
        # The upsampling path reverses the downsampling process
        # It increases spatial resolution while reducing feature channels
        # This helps in refining the spatial details and maintaining model capacity
        self.ups = nn.ModuleList()
        for i in range(len(cnn_units)-1, 0, -1):
            self.ups.append(Up(cnn_units[i], cnn_units[i-1]))
        
        # Output convolutions - match CnnIg architecture
        self.outc = OutConv(cnn_units[0], channel_out)
        self.last_height = OutConv(height_in, height_in)  # Preserve height dimensions
        self.sigmoid = nn.Sigmoid()
        
        # Add dropout if specified
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x, time_cond=None, cond=None):
        """
        Forward pass that supports both standard and diffusion model modes.
        
        Args:
            x: For standard model: tuple of (x3d_norm, x2d_norm, x2d_orig)
               For diffusion model: noisy sample tensor
            time_cond: Time conditioning for diffusion
            cond: Dict with conditioning info for diffusion
            
        Returns:
            Predicted output
        """
        # Case 1: EDM diffusion mode
        if cond is not None and time_cond is not None:
            x3d_norm = cond["x3d_norm"]
            x2d_norm = cond["x2d_norm"]
            x2d_orig = cond["x2d_orig"]
            
            # Process with diffusion-specific forward implementation
            return self._forward_diffusion(x3d_norm, x2d_norm, x2d_orig, x, time_cond)
        
        # Case 2: Standard mode (no diffusion) using tuple input
        elif isinstance(x, tuple) and len(x) == 3:
            x3d_norm, x2d_norm, x2d_orig = x
            return self._forward_standard(x3d_norm, x2d_norm, x2d_orig)
        
        # Case 3: Standard function call with positional arguments
        else:
            return self._forward_standard(x, time_cond, cond)
        

    def _forward_standard(self, x3d_norm, x2d_norm, x2d_orig):
        """
        Standard forward pass without diffusion components
        """
        # Broadcast 2D data to match height dimension of 3D data
        x2d = x2d_norm.unsqueeze(1).repeat(1, x3d_norm.shape[1], 1)  # [batch, height, features_2d]
        x3d_norm = torch.cat([x2d, x3d_norm], dim=-1)  # [batch, height, features_2d + features_3d]
        
        # Create extra 2d nodes with ones and concatenate with the 3d column
        ones = torch.ones(x3d_norm.shape[0], 1, x3d_norm.shape[-1], device=x3d_norm.device)
        x = torch.cat([ones, x3d_norm], dim=1)
        
        # Apply dropout if specified
        if self.dropout:
            x = self.dropout(x)
        
        # U-Net encoder-decoder architecture with skip connections saved in xx
        # xx stores the feature maps from each level of the encoder path that will be used as skip connections
        xx = list()
        
        # Convert to [batch, features, height] - PyTorch expects channels first format
        x = x.permute(0, 2, 1)
        
        # Initial convolution
        inp = self.inc(x)
        xx.append(inp)
        
        # Encoder path with skip connections
        for down in self.downs:
            xx.append(down(xx[-1]))
        
        # Decoder path
        # .pop() removes and returns the last element from the list.  
        # The encoder features were added in order from shallow to deep, we retrieve them in the opposite order 
        # - deep to shallow - which is exactly what we need for the decoder path.
        y = xx.pop()
        for up in self.ups:
            y = up(y, xx.pop())
        
        # Output convolutions
        y = self.outc(y).permute(0, 2, 1)  # Back to [batch, height, features]
        y = self.last_height(y)
        
        # Apply sigmoid activation before scaling
        y = self.sigmoid(y)
        
        # Scale output based on physical quantities
        y = self._scale_output(y, x2d_orig)
            
        return y.squeeze()
        
    def _forward_diffusion(self, x3d_norm, x2d_norm, x2d_orig, noisy_sample, time_cond):
        """
        Diffusion model forward pass with time conditioning
        """
        # Broadcast 2D data to match height dimension of 3D data
        x2d = x2d_norm.unsqueeze(1).repeat(1, x3d_norm.shape[1], 1)  # [batch, height, features_2d]
        x3d_norm = torch.cat([x2d, x3d_norm], dim=-1)
        # Create extra 2d nodes with ones and concatenate with the 3d column
        ones = torch.ones(x3d_norm.shape[0], 1, x3d_norm.shape[-1], device=x3d_norm.device)
        x = torch.cat([ones, x3d_norm], dim=1)
        
        # Apply dropout if specified
        if self.dropout:
            x = self.dropout(x)
        
        # U-Net encoder-decoder architecture with skip connections
        xx = list()
        
        # Convert to [batch, features, height]
        x = x.permute(0, 2, 1)
        
        # Process input features
        features = self.inc(x)
        xx.append(features)
        
        # Ensure noisy_sample has the right format
        if noisy_sample.shape[1] != self.channel_out or noisy_sample.shape[2] != self.height_in:
            # Permute if dimensions don't match expected format
            if noisy_sample.shape[2] == self.channel_out and noisy_sample.shape[1] == self.height_in:
                noisy_sample = noisy_sample.permute(0, 2, 1)
        
        # Encoder path with skip connections
        for down in self.downs:
            xx.append(down(xx[-1]))
        
        # Apply time embedding at the bottleneck
        y = xx[-1]
        
        # Process time embedding in the bottleneck:
        #-----------------------------------------------------------------------------------------
        # Why it's necessary: Different noise levels require different denoising behaviors. 
        # At high noise levels (early diffusion steps), the model needs to focus on rough structure; 
        # at low noise levels (late steps), it needs to focus on fine details. 
        # Without time conditioning, the model wouldn't know which denoising behavior to apply.
        t_emb = self.time_embedding(time_cond)
        t_emb = self.time_proj(t_emb)
        
        # Add time embedding to bottleneck features (broadcast across spatial dimensions)
        t_emb = t_emb.unsqueeze(-1).repeat(1, 1, y.shape[-1])
        y = y + t_emb
        
        # Decoder path
        xx.pop()  # Remove the bottleneck features since we've used them
        for up in self.ups:
            y = up(y, xx.pop())
        
        # Output processing
        y = self.outc(y).permute(0, 2, 1)  # Back to [batch, height, features]
        y = self.last_height(y)
        
        # Apply sigmoid activation before scaling
        y = self.sigmoid(y)
        
        # Always scale output
        y = self._scale_output(y, x2d_orig)
            
        return y.squeeze() 