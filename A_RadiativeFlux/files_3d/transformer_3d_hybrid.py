import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from .base_methods import BaseRadiationModel
from .graph_3d_full import get_3d_graph


class VerticalTransformerLayer(nn.Module):
    """
    Standard transformer attention applied to the vertical dimension within each column.
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, dim)
        
    def forward(self, x):
        # x shape: [batch, columns, height, dim]
        B, N, H, D = x.shape
        
        # Reshape to process all columns independently: [B*N, H, D]
        x = x.view(B * N, H, D)
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B * N, H, self.heads, -1).transpose(1, 2), qkv)
        
        # Apply attention along height dimension
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B * N, H, -1)
        out = self.to_out(out)
        
        # Reshape back: [B, N, H, D]
        return out.view(B, N, H, D)


class HorizontalMessagePassing(MessagePassing):
    """
    GNN-style message passing for horizontal interactions between neighboring columns.
    """
    def __init__(self, dim, dropout=0.0):
        super().__init__(aggr='mean')  # Use mean aggregation
        
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        
    def forward(self, x, edge_index):
        # x shape: [batch*columns*height, dim]
        # Process horizontal edges only
        return self.propagate(edge_index, x=x)
    
    def message(self, x_i, x_j):
        # x_i: receiver nodes, x_j: sender nodes
        return self.message_mlp(torch.cat([x_i, x_j], dim=-1))
    
    def update(self, aggr_out, x):
        # Combine aggregated messages with node features
        return self.update_mlp(torch.cat([x, aggr_out], dim=-1))


class HybridTransformerGNNLayer(nn.Module):
    """
    Hybrid layer that combines vertical transformer attention with horizontal message passing.
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        self.vertical_transformer = VerticalTransformerLayer(dim, heads, dim_head, dropout)
        self.horizontal_gnn = HorizontalMessagePassing(dim, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def extract_horizontal_edges(self, edge_index, num_columns, num_height_levels, batch_size):
        """Extract edges that connect nodes at the same height level (horizontal edges)."""
        horizontal_edges = []
        
        for i in range(edge_index.size(1)):
            src, dst = edge_index[:, i]
            
            # Convert to column and height indices
            src_col = src % num_columns
            dst_col = dst % num_columns
            src_height = (src // num_columns) % num_height_levels
            dst_height = (dst // num_columns) % num_height_levels
            
            # Only keep edges within the same height level and between different columns
            if src_height == dst_height and src_col != dst_col:
                horizontal_edges.append([src.item(), dst.item()])
        
        if horizontal_edges:
            return torch.tensor(horizontal_edges, device=edge_index.device).T
        else:
            # Return empty edge index if no horizontal edges
            return torch.zeros((2, 0), dtype=torch.long, device=edge_index.device)
    
    def forward(self, x, edge_index, num_columns, num_height_levels):
        B, N, H, D = x.shape
        
        # 1. Vertical transformer processing
        x_vertical = self.vertical_transformer(self.norm1(x))
        x = x + x_vertical
        
        # 2. Horizontal message passing
        # Extract horizontal edges
        horizontal_edge_index = self.extract_horizontal_edges(
            edge_index, num_columns, num_height_levels, B
        )
        
        if horizontal_edge_index.size(1) > 0:
            # Reshape for message passing: [B*N*H, D]
            x_flat = x.view(B * N * H, D)
            
            # Apply horizontal message passing
            x_horizontal = self.horizontal_gnn(self.norm2(x_flat), horizontal_edge_index)
            
            # Reshape back and add residual
            x_horizontal = x_horizontal.view(B, N, H, D)
            x = x + x_horizontal
        
        return x


class HybridTransformer3D(BaseRadiationModel):
    """
    Hybrid 3D Transformer that combines vertical transformer attention 
    with horizontal GNN message passing for triangular ICON grid data.
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
        
        # Hybrid transformer-GNN layers
        self.hybrid_layers = nn.ModuleList([
            HybridTransformerGNNLayer(embed_dim, heads, dim_head, dropout)
            for _ in range(depth)
        ])
        
        # Feed-forward layers
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, mlp_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, embed_dim),
                nn.Dropout(dropout)
            )
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
        
        # Apply hybrid transformer-GNN layers with feed-forward
        for hybrid_layer, mlp_layer in zip(self.hybrid_layers, self.mlp_layers):
            # Hybrid attention (vertical transformer + horizontal GNN)
            x = hybrid_layer(x, edge_index, num_columns, self.num_height_levels)
            
            # Feed-forward with residual
            x = x + mlp_layer(x)
        
        # Final processing
        x = self.norm(x)
        x = self.output_proj(x)
        x = self.sigmoid(x)
        
        # Scale output
        output = self._scale_output(x, x2d_orig)
        
        return output 