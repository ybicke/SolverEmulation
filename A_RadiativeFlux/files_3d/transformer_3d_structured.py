import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_methods import BaseRadiationModel
from .graph_3d_full import get_3d_graph

class StructuredAttention(nn.Module):
    """
    Structured attention mechanism that handles both vertical and horizontal attention
    while respecting the ICON grid structure.
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        
        # Separate projections for vertical and horizontal attention
        self.to_qkv_vertical = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_qkv_horizontal = nn.Linear(dim, inner_dim * 3, bias=False)
        
        self.to_out = nn.Linear(inner_dim * 2, dim)  # *2 because we concatenate vertical and horizontal
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, num_columns):
        """
        x: [batch, columns, height, dim]
        edge_index: adjacency information from ICON grid
        """
        B, N, H, D = x.shape
        
        # 1. Vertical Attention (within columns)
        # Reshape for vertical attention: [B*N, H, D]
        x_vert = x.view(B * N, H, D)
        qkv_vert = self.to_qkv_vertical(x_vert).chunk(3, dim=-1)
        q_vert, k_vert, v_vert = map(lambda t: t.view(B * N, H, self.heads, -1).transpose(1, 2), qkv_vert)
        
        # Compute vertical attention
        dots_vert = torch.matmul(q_vert, k_vert.transpose(-1, -2)) * self.scale
        attn_vert = F.softmax(dots_vert, dim=-1)
        attn_vert = self.dropout(attn_vert)
        out_vert = torch.matmul(attn_vert, v_vert)
        out_vert = out_vert.transpose(1, 2).contiguous().view(B, N, H, -1)
        
        # 2. Horizontal Attention (between columns at each height level)
        # Create attention mask based on ICON grid structure
        mask = torch.zeros(N, N, device=x.device, dtype=torch.bool)
        edge_src, edge_dst = edge_index
        
        # Extract unique column connections
        for i in range(edge_index.size(1)):
            src_col = edge_src[i] % num_columns
            dst_col = edge_dst[i] % num_columns
            src_height = edge_src[i] // num_columns
            dst_height = edge_dst[i] // num_columns
            
            # Only connect columns at same height
            if src_height == dst_height:
                mask[src_col, dst_col] = True
        
        # Process each height level separately
        out_horiz = []
        for h in range(H):
            x_h = x[:, :, h]  # [B, N, D]
            
            # Project to Q, K, V
            qkv_h = self.to_qkv_horizontal(x_h).chunk(3, dim=-1)
            q_h, k_h, v_h = map(lambda t: t.view(B, N, self.heads, -1).transpose(1, 2), qkv_h)
            
            # Compute attention scores
            dots_h = torch.matmul(q_h, k_h.transpose(-1, -2)) * self.scale
            
            # Apply ICON grid structure mask
            dots_h = dots_h.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
            attn_h = F.softmax(dots_h, dim=-1)
            attn_h = self.dropout(attn_h)
            
            out_h = torch.matmul(attn_h, v_h)
            out_h = out_h.transpose(1, 2).contiguous().view(B, N, -1)
            out_horiz.append(out_h)
        
        out_horiz = torch.stack(out_horiz, dim=2)  # [B, N, H, D]
        
        # Combine vertical and horizontal attention
        out = torch.cat([out_vert, out_horiz], dim=-1)
        return self.to_out(out)


class StructuredTransformerBlock(nn.Module):
    """Transformer block with structured attention."""
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = StructuredAttention(dim, heads, dim_head, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, edge_index, num_columns):
        x = x + self.attn(self.norm1(x), edge_index, num_columns)
        x = x + self.mlp(self.norm2(x))
        return x


class StructuredTransformer3D(BaseRadiationModel):
    """
    Pure transformer model for 3D atmospheric data that respects ICON grid structure.
    Uses structured attention patterns for both vertical and horizontal interactions.
    """
    def __init__(self,
                 total_cols,
                 grid_file_path,
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
                 fully_connected=False,
                 disable_horizontal=False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.device = device
        self.channels_out = channels_out
        self.embed_dim = embed_dim
        
        # Store grid parameters
        self.grid_file_path = grid_file_path
        self.triangle_id = triangle_id
        self.num_height_levels = num_height_levels
        self.division_factor = division_factor
        self.total_cols = total_cols
        self.fully_connected = fully_connected
        self.disable_horizontal = disable_horizontal
        
        # Input projection
        total_channels = channels_in_3d + channels_in_2d
        self.input_proj = nn.Linear(total_channels, embed_dim)
        
        # Position embeddings
        self.pos_embedding_vertical = nn.Parameter(torch.randn(1, 1, num_height_levels, embed_dim))
        self.pos_embedding_horizontal = nn.Parameter(torch.randn(1, total_cols, 1, embed_dim))
        
        # Transformer blocks
        mlp_dim = int(embed_dim * mlp_ratio)
        self.transformer_blocks = nn.ModuleList([
            StructuredTransformerBlock(embed_dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])
        
        # Output projection
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, channels_out)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x3d_norm, x2d_norm, x2d_orig):
        B, N, L, _ = x3d_norm.shape
        
        # Get graph structure
        edge_index, num_columns = get_3d_graph(
            grid_file_path=self.grid_file_path,
            triangle_id=self.triangle_id,
            num_height_levels=self.num_height_levels,
            batch_size=B,
            total_cols=self.total_cols,
            division_factor=self.division_factor,
            device=self.device,
            fully_connected=self.fully_connected,
            disable_horizontal=self.disable_horizontal
        )
        
        # Prepare features: broadcast 2D to all levels and concatenate
        x2d_repeated = x2d_norm.unsqueeze(2).repeat(1, 1, L, 1)
        x_concat = torch.cat([x3d_norm, x2d_repeated], dim=-1)
        
        # Project to embedding space: [B, N, L, embed_dim]
        x = self.input_proj(x_concat)
        
        # Add position embeddings
        x = x + self.pos_embedding_vertical + self.pos_embedding_horizontal
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, edge_index, num_columns)
        
        # Final processing
        x = self.norm(x)
        x = self.output_proj(x)
        x = self.sigmoid(x)
        
        # Scale output
        output = self._scale_output(x, x2d_orig)
        
        return output 