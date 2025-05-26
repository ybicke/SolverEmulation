import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_methods import BaseRadiationModel
from .graph_3d_full import get_3d_graph
import numpy as np


class LocalPatchAttention(nn.Module):
    """
    Local patch attention that creates neighborhoods around each column 
    and applies standard transformer attention within these patches.
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, max_neighbors=7):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.max_neighbors = max_neighbors
        inner_dim = dim_head * heads
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, dim)
        
    def create_local_patches(self, x, edge_index, num_columns):
        """
        Create local patches around each column using the graph structure.
        Returns patches and attention masks.
        """
        B, N, H, D = x.shape
        device = x.device
        
        # Create adjacency matrix from edge_index
        adj_matrix = torch.zeros(num_columns, num_columns, device=device)
        
        # Extract horizontal edges (same height level)
        col_edges = []
        for i in range(0, edge_index.size(1), B):  # Account for batching
            src, dst = edge_index[:, i]
            src_col = src % num_columns
            dst_col = dst % num_columns
            src_height = src // num_columns
            dst_height = dst // num_columns
            
            # Only consider edges within the same height level
            if src_height == dst_height:
                col_edges.append([src_col.item(), dst_col.item()])
        
        # Build adjacency for columns
        for src_col, dst_col in col_edges:
            adj_matrix[src_col, dst_col] = 1
            
        # Add self-connections
        adj_matrix.fill_diagonal_(1)
        
        # Create patches: for each column, include itself and neighbors
        patches = []
        masks = []
        
        for col_idx in range(num_columns):
            # Find neighbors
            neighbors = torch.nonzero(adj_matrix[col_idx]).squeeze(-1)
            
            # Pad or truncate to max_neighbors
            if len(neighbors) > self.max_neighbors:
                neighbors = neighbors[:self.max_neighbors]
            
            patch_indices = torch.zeros(self.max_neighbors, dtype=torch.long, device=device)
            patch_mask = torch.zeros(self.max_neighbors, dtype=torch.bool, device=device)
            
            patch_indices[:len(neighbors)] = neighbors
            patch_mask[:len(neighbors)] = True
            
            # Extract patch data: [B, max_neighbors, H, D]
            patch_data = x[:, patch_indices]
            
            patches.append(patch_data)
            masks.append(patch_mask)
            
        return torch.stack(patches, dim=1), torch.stack(masks, dim=1)
    
    def forward(self, x, edge_index, num_columns):
        B, N, H, D = x.shape
        
        # Create local patches: [B, num_columns, max_neighbors, H, D]
        patches, masks = self.create_local_patches(x, edge_index, num_columns)
        
        # Process each height level separately
        outputs = []
        for h in range(H):
            # Extract height slice: [B, num_columns, max_neighbors, D]
            x_h = patches[:, :, :, h, :]
            mask_h = masks  # [num_columns, max_neighbors]
            
            # Reshape for attention: [B*num_columns, max_neighbors, D]
            x_h = x_h.view(B * N, self.max_neighbors, D)
            
            # Apply attention within each patch
            qkv = self.to_qkv(x_h).chunk(3, dim=-1)
            q, k, v = map(lambda t: t.view(B * N, self.max_neighbors, self.heads, -1).transpose(1, 2), qkv)
            
            # Attention scores
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            
            # Apply attention mask
            mask_expanded = mask_h.unsqueeze(0).unsqueeze(0).expand(B * N, self.heads, -1, -1)
            dots = dots.masked_fill(~mask_expanded, float('-inf'))
            
            attn = F.softmax(dots, dim=-1)
            attn = self.dropout(attn)
            
            # Apply attention to values
            out_h = torch.matmul(attn, v)
            out_h = out_h.transpose(1, 2).contiguous().view(B * N, self.max_neighbors, -1)
            out_h = self.to_out(out_h)
            
            # Take the center column output (index 0 in each patch)
            center_out = out_h[:, 0, :].view(B, N, D)
            outputs.append(center_out)
            
        # Recombine height levels: [B, N, H, D]
        return torch.stack(outputs, dim=2)


class PatchTransformerBlock(nn.Module):
    """Transformer block with local patch attention."""
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0, max_neighbors=7):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.patch_attn = LocalPatchAttention(dim, heads, dim_head, dropout, max_neighbors)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, edge_index, num_columns):
        x = x + self.patch_attn(self.norm1(x), edge_index, num_columns)
        x = x + self.mlp(self.norm2(x))
        return x


class PatchTransformer3D(BaseRadiationModel):
    """
    3D Transformer using local patch attention for triangular ICON grid data.
    Creates neighborhoods around each column and applies attention within patches.
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
                 max_neighbors=7,
                 fully_connected=False,
                 disable_horizontal=False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.device = device
        self.channels_out = channels_out
        self.embed_dim = embed_dim
        
        # Store graph parameters
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
        
        # Transformer blocks
        mlp_dim = int(embed_dim * mlp_ratio)
        self.transformer_blocks = nn.ModuleList([
            PatchTransformerBlock(embed_dim, heads, dim_head, mlp_dim, dropout, max_neighbors)
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
        
        # Apply patch transformer blocks
        for block in self.transformer_blocks:
            x = block(x, edge_index, num_columns)
        
        # Final processing
        x = self.norm(x)
        x = self.output_proj(x)
        x = self.sigmoid(x)
        
        # Scale output
        output = self._scale_output(x, x2d_orig)
        
        return output 