# Adopted from https://github.com/zongyi-li/fourier_neural_operator/blob/main/fourier_1d.py
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.fft
from einops.layers.torch import Rearrange


from utils.base_methods import BaseRadiationModel



class AFNO1D(nn.Module):
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    hidden_size_factor: expansion factor for the intermediate FFT dimension
    """
    def __init__(self, dim, num_blocks, sparsity_threshold, hard_thresholding_fraction, hidden_size_factor):
        super().__init__()
        assert dim % num_blocks == 0, f"dim {dim} should be divisible by num_blocks {num_blocks}"

        self.dim = dim
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.dim // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, N, C = x.shape

        x = torch.fft.rfft(x, dim=1, norm="ortho")
        x = x.reshape(B, N // 2 + 1, self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, N // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, N // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = N // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

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

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, N // 2 + 1, C)
        x = torch.fft.irfft(x, n=N, dim=1, norm="ortho")
        x = x.type(dtype)
        return x + bias
    
    
    
class Mlp(nn.Module):
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
    def __init__(self, 
                 dim,
                 mlp_ratio,
                 drop,
                 act_layer,
                 norm_layer,
                 fno_blocks,
                 sparsity_threshold,
                 hard_thresholding_fraction,
                 hidden_size_factor,
                 ):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.filter = AFNO1D(dim=dim,
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

        x = x + residual
        residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        return x
    
    
class AFNONet(BaseRadiationModel):
    """
    Args:
        patch_size (int, tuple): patch size
        in_chans (int): number of input channels
        embed_dim (int): embedding dimension
        depth (int): depth of transformer layers, here blocks
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim
        dropout (float): dropout rate
        channel_3d (int): number of 3D channel inputs
        channel_2d (int): number of 2D channel inputs
        channel_out (int): number of output channels
        height_in (int): input height
        fno_blocks (int): number of blocks in AFNO layer
        sparsity_threshold (float): lambda for softshrink in AFNO
        hard_thresholding_fraction (float): fraction of frequencies to keep
        hidden_size_factor (float): expansion factor for FFT dimension
    """
        
    def __init__(self, 
                 patch_size,
                 embed_dim,
                 depth,
                 dropout,
                 channel_3d,
                 channel_2d,
                 channel_out,
                 height_in,
                 mlp_ratio,
                 fno_blocks,
                 sparsity_threshold,
                 hard_thresholding_fraction,
                 hidden_size_factor,
                 device,
                 *args,
                 **kwargs): 

        super().__init__(*args, **kwargs)
       
        self.num_patches = int(height_in/patch_size)
        self.height = height_in
        self.channels_out = channel_out

        channels_in = channel_3d + channel_2d
        patch_dim = channels_in * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (h p) f -> b h p f', p=patch_size),
            nn.Flatten(-2, -1),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        
        # self.dummy_vector = nn.Parameter(torch.randn(1, 1, channel_2d))
        
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(embed_dim)              
        
        self.mlp_head = nn.Linear(embed_dim, patch_size*channel_out)
        self.sigmoid = nn.Sigmoid()
    
        # Create afno blocks
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
                hidden_size_factor=hidden_size_factor,
                )
            for i in range(depth)
        ])
        
        
            
    def forward(self, x3d_norm, x2d_norm, x2d_orig):
        
        # Repeat x2d along the height dimension to match the shape of x3d
        x2d_repeated = x2d_norm.unsqueeze(1).repeat(1, x3d_norm.shape[1], 1)
        x_concat = torch.cat((x3d_norm, x2d_repeated), dim=-1)
        
        # Repeat the dummy vector along the batch dimension, same random nr accross the batch
        #dummy_vector_repeated = self.dummy_vector.repeat(x3d_norm.shape[0], 1, 1)
        # concat_with_x2d = torch.cat((dummy_vector_repeated, x2d_norm.unsqueeze(1)), dim=-1)
        
        ones = torch.ones(x2d_norm.shape[0], 1, x2d_norm.shape[1], device=x3d_norm.device)
        concat_ones_with_x2d = torch.cat((ones, x2d_norm.unsqueeze(1)), dim=-1)
        
        #dummy_vector_repeated = self.dummy_vector.repeat(x3d_norm.shape[0], 1, 1)
        # concat_with_x2d = torch.cat((dummy_vector_repeated, x2d_norm.unsqueeze(1)), dim=-1)
        
        # Concatenate the resulting tensor as an additional height level
        x_concat = torch.cat((concat_ones_with_x2d, x_concat), dim=1)
        x = self.to_patch_embedding(x_concat)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)   
        
        x = self.mlp_head(x)
        x = self.sigmoid(x)
        x = self._scale_output(x, x2d_orig)
        
        return x.squeeze()    
    
    
    
