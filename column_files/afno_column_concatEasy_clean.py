import math
import logging
from functools import partial
from collections import OrderedDict
from copy import Error, deepcopy
from re import S
from numpy.lib.arraypad import pad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft
from torch.nn.modules.container import Sequential
# from main_afnonet import get_args
from torch.utils.checkpoint import checkpoint_sequential

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


from afno.afno1d import AFNO1D
# from afno.afno2d import AFNO2D
from afno.bfno2d import BFNO2D
from afno.ls import AttentionLS
from afno.sa import SelfAttention
from afno.gfn import GlobalFilter

import matplotlib.pyplot as plt
import os

_logger = logging.getLogger(__name__)   


    
# this class is borrowed from vit_column, here for normalizing the input
class Normalization(nn.Module):
    """ Normalize the input based on mean and std """    
    def __init__(self, std, mean, axis=None):
        super().__init__()
        self.std = std
        self.mean = mean
        self.axis = axis

    def forward(self, x):
        # TODO: only supports normalization on last dim
        # should be extended to custom dims
        assert x.size(dim=-1) == self.std.size(dim=0), \
            f'Dimension mismatch ( {x.size(dim=-1)} != {self.std.size(dim=0)})'
        return (x - self.mean)/self.std
    
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
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
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 h=14, # I do overwrite them
                 w=8,
                 mixing_type="afno",
                 hidden_size=256,
                 fno_blocks=8,
                 sparsity_threshold=0.01,
                 hard_thresholding_fraction=1.0,
                 hidden_size_factor=1,
                 double_skip=True
                 ):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        # could potentially implment other mixing types here such as bfno, sa from the paper
        # here hidden_size = hidden_size before, for making embedding dimension smaller..? 
        # I had to adjust it to make it suit the division by the block size in the afno1D
        if mixing_type == "afno":
            self.filter = AFNO1D(hidden_size=dim,
                                 num_blocks=fno_blocks,
                                 sparsity_threshold=sparsity_threshold,
                                 hard_thresholding_fraction=hard_thresholding_fraction,
                                 hidden_size_factor=1)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.double_skip = double_skip

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x
    
    
class AFNONet(nn.Module):
    """
    Args:
        patch_size (int, tuple): patch size
        in_chans (int): number of input channels
        embed_dim (int): embedding dimension
        depth (int): depth of transformer layers, here blocks
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim
        drop_rate (float): dropout rate
        drop_path_rate (float): stochastic depth rate
        norm_layer: (nn.Module): normalization layer
    """
        
    def __init__(self, 
                 patch_size,
                 num_cells,
                 embed_dim, # (mlp_dim)
                 depth, # actually num of blocks, what about the layers?
                 # heads, not used in afno
                 dropout,
                 emb_dropout=0.,
                 channels_in=12,
                 channels_out=4,
                 height=71,
                 swflx_idx=[2, 3],
                 lwflx_idx=[0, 1], 
                 cosmu0_idx=1, 
                 tsfctrad_idx=5, 
                 mean2d=None,
                 var2d=None, 
                 mean3d=None, 
                 var3d=None,
                 device=None,
                 uniform_drop=False, 
                 drop_path_rate=0.,
                 mlp_ratio=4.,
                 hard_thresholding_fraction=1,
                 sparsity_threshold=0.01,
                 *args,
                 **kwargs): 

        super().__init__()
       
       
        self.num_patches = int(height/patch_size)
        self.num_cells = num_cells
        self.height = height
        self.channels_out = channels_out
        self.swflx_idx = swflx_idx
        self.lwflx_idx = lwflx_idx
        self.cosmu0_idx = cosmu0_idx
        self.tsfctrad_idx = tsfctrad_idx
        
        # Initialize normalizers for 2D and 3D inputs
        self.normalizer2d = Normalization(std=torch.sqrt(var2d), mean=mean2d)
        self.normalizer3d = Normalization(std=torch.sqrt(var3d), mean=mean3d)

        patch_dim = channels_in  * patch_size

        # same patch embedding as in ViT code
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (h p) f -> b h p f', p=patch_size),
            nn.Flatten(-2, -1),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # not needed in lazy apporach
        #self.to_patch_embedding_2D = nn.Sequential(
        #    nn.Linear(channels_in, embed_dim),
        #    nn.LayerNorm(embed_dim)
        #)
        
        self.dummy_vector = nn.Parameter(torch.randn(1, 1, 6))
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(embed_dim)              
        
        # Define the MLP head for final output 
        self.mlp_head = nn.Linear(embed_dim, patch_size*channels_out)
        self.sigmoid = nn.Sigmoid()

      
        # With uniform fals and drop_path_rate to 0 not really used. Responsible for calculating the drop path rates for each 
        # "transformer" block based on the uniform_drop flag and the drop_path_rate value. If uniform_drop is True, the same 
        # drop_path_rate is used for all blocks. Otherwise, a linearly increasing drop path rate is used, starting from 0 and 
        # reaching drop_path_rate at the last block.

        if uniform_drop:
            print('using uniform droppath with expect rate', drop_path_rate)
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            print('using linear droppath with expect rate', drop_path_rate * 0.5)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        
          
        h=height // patch_size
        w=1
        
      
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, 
                mlp_ratio=mlp_ratio,
                drop=dropout,
                drop_path=dpr[i],
                norm_layer=nn.LayerNorm,
                h=h,
                w=w,
                sparsity_threshold=sparsity_threshold,
                hard_thresholding_fraction = hard_thresholding_fraction)
                for i in range(depth)
                
        ])
        

    # Radiation task specifics:
    def _unscale_swflx(self, swflx, cosmu0):
        return torch.where(
            cosmu0 >= torch.tensor(1e-4, dtype=torch.float32),
            swflx * (cosmu0 * 1400),
            0
        )

    def _unscale_lwflx(self, lwflx, tsfctrad):
        stefan_boltzmann_const = torch.tensor(5.670374419e-08, dtype=torch.float32)
        return torch.where(
            tsfctrad >= torch.tensor(1e-4, dtype=torch.float32),
            lwflx * torch.pow(tsfctrad, 4) * stefan_boltzmann_const,
            lwflx
        )

    def _scale_output(self, y_pred, x2d):
        y_pred_scaled = []

        for i in range(y_pred.shape[-1]):
            f_pred = y_pred[..., i:i+1]

            if i in self.swflx_idx:
                cosmu0 = x2d[..., self.cosmu0_idx]
                cosmu0 = torch.tile(
                    cosmu0[..., None, None], 
                    (1, 1, f_pred.shape[-2], 1)
                )
                y_pred_scaled.append(self._unscale_swflx(f_pred, cosmu0))
            elif i in self.lwflx_idx:
                tsfctrad = x2d[..., self.tsfctrad_idx]
                tsfctrad = torch.tile(
                    tsfctrad[..., None, None], 
                    (1, 1, f_pred.shape[-2], 1)
                )
                y_pred_scaled.append(self._unscale_lwflx(f_pred, tsfctrad))
            else:
                y_pred_scaled.append(f_pred)

        y_pred = torch.cat(y_pred_scaled, dim=-1)
        return y_pred
        
            

            
    def forward_features(self, x3d, x2d):

        x3d = self.normalizer3d(x3d)
        x2d = self.normalizer2d(x2d)
        
        # Repeat x2d along the height dimension to match the shape of x3d
        x2d_repeated = x2d.unsqueeze(1).repeat(1, x3d.shape[1], 1)
        x_concat = torch.cat((x3d, x2d_repeated), dim=-1)
        
        # Repeat the dummy vector along the batch dimension, same random nr accross the batch
        dummy_vector_repeated = self.dummy_vector.repeat(x3d.shape[0], 1, 1)
        concat_with_x2d = torch.cat((dummy_vector_repeated, x2d.unsqueeze(1)), dim=-1)
        
        # Concatenate the resulting tensor as an additional height level
        x_concat = torch.cat((x_concat, concat_with_x2d), dim=1)
        x = self.to_patch_embedding(x_concat)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x3d, x2d):
        x = self.forward_features(x3d, x2d)
        
        x = self.mlp_head(x)
        x = self.sigmoid(x)
        x = self._scale_output(x, x2d)
        
        return x.squeeze()    
    
    
    
