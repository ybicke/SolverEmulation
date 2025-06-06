# Code parts used from https://github.com/NVlabs/AFNO-pytorch
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from einops.layers.torch import Rearrange

_logger = logging.getLogger(__name__)


class AFNO1D(nn.Module):
    """
    1D Adaptive Fourier Neural Operator
    
    Args:
        hidden_size: channel dimension size
        num_blocks: how many blocks to use in the block diagonal weight matrices 
                   (higher => less complexity but less parameters)
        sparsity_threshold: lambda for softshrink
        hard_thresholding_fraction: how many frequencies you want to completely mask out
                                   (lower => hard_thresholding_fraction^2 less FLOPs)
        hidden_size_factor: factor to scale hidden dimension in MLP
    """
    def __init__(self, 
                 hidden_size,
                 num_blocks,
                 sparsity_threshold,
                 hard_thresholding_fraction,
                 hidden_size_factor):
        super().__init__()
        assert hidden_size % num_blocks == 0, \
            f"hidden_size {hidden_size} should be divisible by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        
        # The AFNO1D layer uses a block diagonal matrix structure for its weight matrices. 
        # The weight matrices are divided into smaller blocks along the diagonal.
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        # Preparation for blockwise processing: 2 for real and imag, hidden_size_factor for output
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size,  self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        
        
    def forward(self, x):
        bias = x
        dtype = x.dtype
        x = x.float()
        B, N, C = x.shape

        # Step 1: Token Mixing (Discrete Fourier Transform)
        x = torch.fft.rfft(x, dim=1, norm="ortho")
        x = x.reshape(B, N // 2 + 1, self.num_blocks, self.block_size)
        
        o1_real = torch.zeros([B, N // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, N // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = N // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        # Step 2: Channel Mixing - blockwise spatial mixing/processing
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

        # Step 3: Token Demixing (Inverse Discrete Fourier Transform)
        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, N // 2 + 1, C)        
        x = torch.fft.irfft(x, n=N, dim=1, norm="ortho")
        
        x = x.type(dtype)
        return x + bias


class Normalization(nn.Module):
    """Normalize the input based on mean and std"""    
    def __init__(self, std, mean, axis=None):
        super().__init__()
        self.std = std
        self.mean = mean
        self.axis = axis

    def forward(self, x):
        assert x.size(dim=-1) == self.std.size(dim=0), \
            f'Dimension mismatch ({x.size(dim=-1)} != {self.std.size(dim=0)})'
        return (x - self.mean) / self.std
    

class Mlp(nn.Module):
    """Multi-layer perceptron"""
    def __init__(self, in_features, hidden_features, out_features, act_layer, drop):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """AFNO Block with normalization, filtering, and MLP"""
    def __init__(self, 
                 dim,
                 mlp_ratio,
                 drop,
                 act_layer,
                 norm_layer,
                 fno_blocks,
                 sparsity_threshold,
                 hard_thresholding_fraction,
                 hidden_size_factor):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        # AFNO filter for frequency domain processing
        self.filter = AFNO1D(hidden_size=dim,
                            num_blocks=fno_blocks,
                            sparsity_threshold=sparsity_threshold,
                            hard_thresholding_fraction=hard_thresholding_fraction,
                            hidden_size_factor=hidden_size_factor)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, 
                      hidden_features=mlp_hidden_dim,
                      out_features=dim,
                      act_layer=act_layer, 
                      drop=drop)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        # Default double skip connection
        x = x + residual
        residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        return x
    

class AFNONet(nn.Module):
    """
    Adaptive Fourier Neural Operator Network
    
    Args:
        patch_size (int): patch size
        num_cells (int): number of cells
        embed_dim (int): embedding dimension
        depth (int): depth of transformer layers (number of blocks)
        dropout (float): dropout rate for MLP and positional embeddings
        emb_dropout (float): dropout rate for patch embeddings
        channels_in (int): number of input channels for 3D data
        channels_in_2D (int): number of input channels for 2D data
        channels_out (int): number of output channels
        height (int): height dimension
        mlp_ratio (float): ratio of mlp hidden dim to embedding dim
        hard_thresholding_fraction (float): fraction for hard thresholding
        sparsity_threshold (float): sparsity threshold for soft thresholding
        fno_blocks (int): number of FNO blocks
        hidden_size_factor (int): hidden size factor for AFNO
    """
        
    def __init__(self, 
                 patch_size,
                 num_cells,
                 embed_dim,
                 depth,
                 dropout,
                 emb_dropout,
                 channels_in_3D,
                 channels_in_2D, 
                 channels_out,
                 height,
                 mlp_ratio,
                 hard_thresholding_fraction,
                 sparsity_threshold,
                 fno_blocks,
                 hidden_size_factor,
                 **kwargs):

        super().__init__()
        
        self.num_patches = int(height / patch_size)
        self.num_cells = num_cells
        self.height = height
        self.channels_out = channels_out

        # Calculate patch dimension for merged 3D and 2D data
        patch_dim = (channels_in_3D + channels_in_2D) * patch_size

        # Patch embedding similar to ViT
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (h p) f -> b h p f', p=patch_size),
            nn.Flatten(-2, -1),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.to_patch_embedding_2D = nn.Sequential(
            nn.Linear(channels_in_2D, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.emb_drop = nn.Dropout(p=emb_dropout)
        self.pos_drop = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(embed_dim)              
        
        # Define the MLP head for final output 
        self.mlp_head = nn.Linear(embed_dim, patch_size * channels_out)
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, 
                mlp_ratio=mlp_ratio,
                drop=dropout,
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm,
                fno_blocks=fno_blocks,
                sparsity_threshold=sparsity_threshold,
                hard_thresholding_fraction=hard_thresholding_fraction,
                hidden_size_factor=hidden_size_factor
            )
            for i in range(depth)                
        ])
        


    def forward(self, x3d_norm, x2d_norm, x2d_orig=None):
        """
        Forward pass expecting normalized inputs
        
        Args:
            x3d_norm: normalized 3D input data
            x2d_norm: normalized 2D input data  
            x2d_orig: original 2D data (currently unused)
        """
        # Ensure inputs are float32 to avoid dtype mismatches
        x3d_norm = x3d_norm.float()
        x2d_norm = x2d_norm.float()
        
        # Expand 2D data to match 3D data height dimension
        x2d_to_70 = x2d_norm.unsqueeze(1).repeat(1, x3d_norm.shape[1], 1)
        x3d_merged = torch.cat((x3d_norm, x2d_to_70), dim=-1)
        
        x = self.to_patch_embedding(x3d_merged)
        x = self.emb_drop(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.mlp_head(x)
        
        return x.squeeze()    
    
    
    
