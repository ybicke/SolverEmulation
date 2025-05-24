# Adopted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .base_methods import BaseRadiationModel


    

class FeedForward(nn.Module):
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


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head *  heads
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
    
 
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
  
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x) 
        # Norm after blocks not standard in vanilla ViT

class ViT(BaseRadiationModel):
        
    def __init__(
            self,
             patch_size,
             dim,
             mlp_dim, 
             depth,
             heads,
             channel_3d,
             channel_2d,
             channel_out,
             height_in,
             dim_head,
             dropout,
             emb_dropout,
             device,
             *args, **kwargs
        ):
        
        # Call the base class constructor - flux parameters will be passed via kwargs
        super().__init__(*args, **kwargs)

        # Model architecture setup
        self.num_patches = int(height_in/patch_size)
        self.channels_out = channel_out

        # Total input channels for patch embedding
        channels_in = channel_3d + channel_2d
        patch_dim = channels_in * patch_size
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (h p) f -> b h p f', p=patch_size),
            nn.Flatten(-2, -1),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        
        # For handling 2D input features
        self.dummy_vector = nn.Parameter(torch.randn(1, 1, channel_2d))

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim, device=device))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Linear(dim, patch_size*channel_out)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x3d_norm, x2d_norm, x2d_orig):
        """Forward pass with pre-normalized inputs
        
        Args:
            x3d_norm: Normalized 3D input data
            x2d_norm: Normalized 2D input data
            x2d_orig: Original 2D input data (used for scaling output)
        """
        # Broadcast each 2d feature along the height column
        x2d_repeated = x2d_norm.unsqueeze(1).repeat(1, x3d_norm.shape[1], 1)
        x_concat = torch.cat((x3d_norm, x2d_repeated), dim=-1)
        
        dummy_vector_repeated = self.dummy_vector.repeat(x3d_norm.shape[0], 1, 1)
        concat_with_x2d = torch.cat((dummy_vector_repeated, x2d_norm.unsqueeze(1)), dim=-1)
        
        x_concat = torch.cat((concat_with_x2d, x_concat), dim=1)
        x = self.to_patch_embedding(x_concat)
        
        x += self.pos_embedding[:, :self.num_patches]
        x = self.dropout(x)
        
        x = self.transformer(x)

        x = self.mlp_head(x)
        x = self.sigmoid(x)
        
        # Use original data for proper scaling
        x = self._scale_output(x, x2d_orig)
        
        return x.squeeze()
        
    
    



