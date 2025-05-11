"""
U-Net CNN model for radiation flux prediction
Adapted from the original CNN model architecture

This module implements a 1D CNN U-Net model for predicting radiative fluxes
in atmospheric columns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_methods import BaseRadiationModel


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

    def forward(self, x3d_norm, x2d_norm, x2d_orig):
        
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
        xx.append(self.inc(x.permute(0, 2, 1)))  # Convert to [batch, features, height], pytorch expects channel first format
        
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
        
        # This 1×1 convolution performs a final processing step across the height dimension.
        # Last_height probably not needed as we have same number of height levels in input and output
        y = self.outc(y).permute(0, 2, 1)  # Back to [batch, height, features]
        y = self.last_height(y)
        
        # Apply sigmoid activation before scaling (as in CnnIg)
        y = self.sigmoid(y)
        
        # Always scale output (consistent with CnnIg)
        y = self._scale_output(y, x2d_orig)
            
        return y.squeeze() 