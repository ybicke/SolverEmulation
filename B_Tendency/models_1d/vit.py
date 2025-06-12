# Adopted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange


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


class ViT(nn.Module):
    """Vision Transformer adapted for atmospheric tendency prediction
    
    This model processes atmospheric column data using a transformer architecture.
    It handles both 3D (height-varying) and 2D (column-wide) input features.
    """
        
    def __init__(
            self,
            patch_size,
            embed_dim,
            depth,
            heads,
            dropout,
            emb_dropout,
            channels_in_3D,
            channels_in_2D, 
            channels_out,
            height,
            mlp_ratio,
            dim_head=64
        ):
        super().__init__()

        # Model architecture setup
        self.patch_size = patch_size
        self.height = height
        self.channels_out = channels_out
        self.num_patches = int(height / patch_size)
        
        # Calculate MLP dimension
        mlp_dim = int(embed_dim * mlp_ratio)
        
        # Total input channels for patch embedding
        channels_in = channels_in_3D + channels_in_2D
        patch_dim = channels_in * patch_size
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (h p) f -> b h p f', p=patch_size),
            nn.Flatten(-2, -1),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(embed_dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Linear(embed_dim, patch_size * channels_out)
    
    def forward(self, x3d_norm, x2d_norm):
        """Forward pass with pre-normalized inputs
        
        Args:
            x3d_norm: Normalized 3D input data [batch, height, channels_3d]
            x2d_norm: Normalized 2D input data [batch, channels_2d]
            
        Returns:
            Tendency predictions [batch, height, channels_out]
        """
        # Handle data types - ensure float32
        x3d_norm = x3d_norm.float()
        x2d_norm = x2d_norm.float()
                
        # Broadcast 2D features along the height dimension
        x2d_repeated = x2d_norm.unsqueeze(1).repeat(1, x3d_norm.shape[1], 1)
        x_concat = torch.cat((x3d_norm, x2d_repeated), dim=-1)
        
        # Create patches and embed them
        x = self.to_patch_embedding(x_concat)
        
        # Add positional embedding
        x += self.pos_embedding[:, :self.num_patches]
        x = self.dropout(x)
        
        # Apply transformer
        x = self.transformer(x)
        x = self.mlp_head(x)
        
        return x
        
    
    



