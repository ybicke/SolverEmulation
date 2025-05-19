import logging
import torch
import torch.nn as nn
import torch.fft

from einops.layers.torch import Rearrange

from afno1d import AFNO1D


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
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, # maybe check without layernorm??
                 mixing_type="afno",
                 hidden_size=256,
                 fno_blocks=8,
                 sparsity_threshold=0.01,
                 hard_thresholding_fraction=1.0,
                 hidden_size_factor=1,
                 double_skip=True):
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
                                 hidden_size_factor=1
                                 )
        
    
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
                 channels_in=13,
                 channels_in_2D = 3, 
                 channels_out=7,
                 height=70,
                 swflx_idx=[2, 3],
                 lwflx_idx=[0, 1], 
                 cosmu0_idx=1, 
                 tsfctrad_idx=5, 
                 device=None,
                 is_test=False,
                 uniform_drop=False, 
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
        
        patch_dim = channels_in * patch_size

        # same patch embedding as in ViT code
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
        self.pos_drop = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(embed_dim)              
        
        # Define the MLP head for final output 
        self.mlp_head = nn.Linear(embed_dim, patch_size*channels_out)
        self.sigmoid = nn.Sigmoid()


      
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, 
                mlp_ratio=mlp_ratio,
                drop=dropout,
                norm_layer=nn.LayerNorm,
                sparsity_threshold=sparsity_threshold,
                hard_thresholding_fraction = hard_thresholding_fraction
                )
                for i in range(depth)                
        ])
        

            


    def forward(self, x3d_norm, x2d_norm, x2d_orig):
        # Both x3d and x2d should already be normalized
        # This function now expects normalized inputs
        x2d_to_70 = x2d_norm.unsqueeze(1).repeat(1, x3d_norm.shape[1], 1)
        x3d_merged = torch.cat((x3d_norm, x2d_to_70), dim=-1)
        
        x = self.to_patch_embedding(x3d_merged)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.mlp_head(x)
        
        #x = self.sigmoid(x)
        #x = self._scale_output(x, x2d_orig)
        
        return x.squeeze()    
    
    
    
