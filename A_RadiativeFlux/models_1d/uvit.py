"""
U-ViT: U-Net with Vision Transformer bottleneck for radiation flux prediction
Combines CNN spatial processing with transformer attention for long-range dependencies

This module implements a hybrid 1D CNN U-Net with transformer attention in the bottleneck
for predicting radiative fluxes in atmospheric columns.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from einops import rearrange

from utils.base_methods import BaseRadiationModel


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
        x1 = F.pad(x1, [diff//2, diff-diff//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention module for the bottleneck"""
    def __init__(self, 
                 dim, 
                 heads, 
                 dim_head, 
                 dropout):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):
    """Feed forward network for transformer block"""
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer block with attention and feed forward"""
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        if mlp_dim is None:
            mlp_dim = dim * 4
            
        self.attention = MultiHeadAttention(dim, heads, dim_head, dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x):
        x = self.attention(x) + x  # Residual connection
        x = self.ff(x) + x         # Residual connection
        return x


class AttentionBottleneck(nn.Module):
    """Attention bottleneck for U-ViT"""
    def __init__(self, channels, depth, heads, dim_head, dropout):
        super().__init__()
        self.channels = channels
        
        # Project channels to a dimension suitable for attention
        self.proj_in = nn.Linear(channels, channels)
        self.proj_out = nn.Linear(channels, channels)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(channels, heads, dim_head = dim_head, mlp_dim = None, dropout = dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x shape: [batch, channels, height]
        batch_size, channels, height = x.shape
        
        # Convert to sequence format for attention: [batch, height, channels]
        x = x.permute(0, 2, 1)
        
        # Project to attention space
        x = self.proj_in(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Final norm and projection
        x = self.norm(x)
        x = self.proj_out(x)
        
        # Convert back to conv format: [batch, channels, height]
        x = x.permute(0, 2, 1)
        
        return x


class UViT(BaseRadiationModel):
    """U-Net with Vision Transformer bottleneck for radiative flux prediction"""
    
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
            # Attention parameters
            attention_heads,
            attention_dim_head,
            attention_depth,
            attention_dropout,
            **kwargs
        ):
        """
        Initialize the U-ViT model for radiative flux prediction.
        
        Args:
            height_in: Number of height levels in input
            channel_3d: Number of 3D input channels (features per height level)
            channel_2d: Number of 2D input channels (global features)
            channel_out: Number of output channels (default: 4 for [lw_up, lw_dn, sw_up, sw_dn])
            cnn_units: List of number of features in each CNN layer
            kernel_sizes: List of kernel sizes for max pooling at each level
            dropout: Dropout rate for CNN layers
            device: Device to run model on
            attention_heads: Number of attention heads in bottleneck
            attention_dim_head: Dimension per attention head
            attention_depth: Number of transformer blocks in bottleneck
            attention_dropout: Dropout rate for attention layers
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
        self.downs = nn.ModuleList()
        for i in range(0, len(cnn_units)-1):      
            self.downs.append(Down(cnn_units[i], cnn_units[i+1], kernel_sizes[i]))
        
        # Attention bottleneck - operates on the deepest feature map
        self.attention_bottleneck = AttentionBottleneck(
            channels=cnn_units[-1],
            depth=attention_depth,
            heads=attention_heads,
            dim_head=attention_dim_head,
            dropout=attention_dropout
        )
        
        # Upsampling path
        self.ups = nn.ModuleList()
        for i in range(len(cnn_units)-1, 0, -1):
            self.ups.append(Up(cnn_units[i], cnn_units[i-1]))
        
        # Output convolutions
        self.outc = OutConv(cnn_units[0], channel_out)
        self.last_height = OutConv(height_in, height_in)
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
        
        # U-Net encoder path with skip connections
        skip_connections = []
        x = self.inc(x.permute(0, 2, 1))  # Convert to [batch, features, height]
        skip_connections.append(x)
        
        # Encoder path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
        
        # Apply attention in the bottleneck
        x = self.attention_bottleneck(x)
        
        # Decoder path with skip connections
        skip_connections.pop()  # Remove the deepest feature map (we just processed it)
        for up in self.ups:
            skip_connection = skip_connections.pop()
            x = up(x, skip_connection)
        
        # Output processing
        x = self.outc(x).permute(0, 2, 1)  # Back to [batch, height, features]
        x = self.last_height(x)
        
        # Apply sigmoid activation before scaling
        x = self.sigmoid(x)
        
        # Scale output
        x = self._scale_output(x, x2d_orig)
            
        return x.squeeze() 