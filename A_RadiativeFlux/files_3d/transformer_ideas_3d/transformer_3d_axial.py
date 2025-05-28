import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_methods import BaseRadiationModel
from .data_utils import get_triangle_indices


class AxialAttention(nn.Module):
    """
    Axial attention that processes vertical and horizontal dimensions separately.
    For triangular grids, we use standard attention for vertical and local conv for horizontal.
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        
        # Vertical attention (along height dimension)
        self.to_qkv_vertical = nn.Linear(dim, inner_dim * 3, bias=False)
        
        # Horizontal processing (local convolution for neighbor interactions)
        self.horizontal_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, dim)
        
    def forward(self, x):
        # x shape: [batch, columns, height, dim]
        B, N, H, D = x.shape
        
        # 1. Vertical attention within each column
        # Reshape to process all columns in parallel: [B*N, H, D]
        x_vert = x.view(B * N, H, D)
        
        qkv = self.to_qkv_vertical(x_vert).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B * N, H, self.heads, -1).transpose(1, 2), qkv)
        
        # Apply attention along height dimension
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)
        
        out_vert = torch.matmul(attn, v)
        out_vert = out_vert.transpose(1, 2).contiguous().view(B * N, H, -1)
        out_vert = self.to_out(out_vert)
        
        # Reshape back: [B, N, H, D]
        out_vert = out_vert.view(B, N, H, D)
        
        # 2. Horizontal processing via local convolution
        # Process each height level separately: [B, H, N, D] -> [B*H, D, N]
        x_horiz = x.permute(0, 2, 1, 3).contiguous().view(B * H, N, D).transpose(1, 2)
        out_horiz = self.horizontal_conv(x_horiz)
        out_horiz = out_horiz.transpose(1, 2).view(B, H, N, D).permute(0, 2, 1, 3)
        
        # Combine vertical attention and horizontal processing
        return out_vert + out_horiz


class AxialTransformerBlock(nn.Module):
    """Transformer block with axial attention and feed forward."""
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = AxialAttention(dim, heads, dim_head, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class AxialTransformer3D(BaseRadiationModel):
    """
    3D Transformer using axial attention for triangular ICON grid data.
    Processes vertical interactions with attention, horizontal with local operations.
    """
    def __init__(self,
                 total_cols,
                 triangle_id,
                 embed_dim,
                 depth,
                 dropout,
                 channels_in_3d,
                 channels_in_2d,
                 channels_out,
                 num_height_levels,
                 device,
                 division_factor,
                 heads=8,
                 dim_head=64,
                 mlp_ratio=4.0,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.device = device
        self.channels_out = channels_out
        self.embed_dim = embed_dim
        
        # Store graph parameters
        self.triangle_id = triangle_id
        self.num_height_levels = num_height_levels
        self.division_factor = division_factor
        self.total_cols = total_cols
        
        # Get triangle indices
        self.triangle_indices = get_triangle_indices(
            triangle_id, division_factor, total_cols
        )
        self.num_columns = len(self.triangle_indices)
        
        # Input projection
        total_channels = channels_in_3d + channels_in_2d
        self.input_proj = nn.Linear(total_channels, embed_dim)
        
        # Transformer blocks
        mlp_dim = int(embed_dim * mlp_ratio)
        self.transformer_blocks = nn.ModuleList([
            AxialTransformerBlock(embed_dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])
        
        # Output projection
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, channels_out)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x3d_norm, x2d_norm, x2d_orig):
        B, N, L, _ = x3d_norm.shape
        
        # Prepare features: broadcast 2D to all levels and concatenate
        x2d_repeated = x2d_norm.unsqueeze(2).repeat(1, 1, L, 1)
        x_concat = torch.cat([x3d_norm, x2d_repeated], dim=-1)
        
        # Project to embedding space: [B, N, L, embed_dim]
        x = self.input_proj(x_concat)
        
        # Apply axial transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Final processing
        x = self.norm(x)
        x = self.output_proj(x)
        x = self.sigmoid(x)
        
        # Scale output
        output = self._scale_output(x, x2d_orig)
        
        return output 