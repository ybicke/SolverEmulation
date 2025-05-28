import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_methods import BaseRadiationModel
from .graph_3d_full import get_3d_graph
import math


class GraphMultiHeadAttention(nn.Module):
    """
    Graph-based multi-head attention that operates on neighborhoods defined by the ICON grid.
    Inspired by GenCast's neighborhood-based self-attention.
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        
        # Projections for queries, keys, values
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        
        # Edge feature processing (for relative position encoding)
        self.edge_proj = nn.Linear(3, heads, bias=False)  # 3D relative positions
        
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, dim)
        
    def create_neighborhood_structure(self, edge_index, num_nodes, device):
        """
        Create neighborhood structure from edge_index for efficient attention computation.
        Returns adjacency list and relative position encodings.
        """
        # Create adjacency list for each node
        adj_list = [[] for _ in range(num_nodes)]
        edge_features = []
        
        for i in range(edge_index.size(1)):
            src, dst = edge_index[:, i]
            src_item, dst_item = src.item(), dst.item()
            
            # Add to adjacency list
            adj_list[src_item].append(dst_item)
            
            # Compute relative position (simplified for ICON grid)
            # In practice, you'd use actual 3D coordinates from the grid file
            rel_pos = torch.tensor([
                (dst_item % 100) - (src_item % 100),  # x-like coordinate
                (dst_item // 100) - (src_item // 100),  # y-like coordinate
                0.0  # z-coordinate (same height level)
            ], device=device)
            edge_features.append(rel_pos)
        
        return adj_list, torch.stack(edge_features) if edge_features else torch.zeros((0, 3), device=device)
    
    def forward(self, x, edge_index, num_columns):
        """
        x: [batch, nodes, dim] - flattened across height levels
        edge_index: graph connectivity
        """
        B, N, D = x.shape
        
        # Project to Q, K, V
        q = self.to_q(x).view(B, N, self.heads, self.dim_head).transpose(1, 2)  # [B, heads, N, dim_head]
        k = self.to_k(x).view(B, N, self.heads, self.dim_head).transpose(1, 2)
        v = self.to_v(x).view(B, N, self.heads, self.dim_head).transpose(1, 2)
        
        # Create neighborhood structure
        adj_list, edge_features = self.create_neighborhood_structure(edge_index, N, x.device)
        
        # Compute attention for each node based on its neighborhood
        attention_outputs = []
        
        for node_idx in range(N):
            neighbors = adj_list[node_idx] if node_idx < len(adj_list) else []
            
            if not neighbors:
                # If no neighbors, just use self-attention
                neighbors = [node_idx]
            
            # Add self-connection if not already present
            if node_idx not in neighbors:
                neighbors = [node_idx] + neighbors
            
            # Extract queries for current node and keys/values for neighbors
            q_node = q[:, :, node_idx:node_idx+1, :]  # [B, heads, 1, dim_head]
            k_neighbors = k[:, :, neighbors, :]  # [B, heads, num_neighbors, dim_head]
            v_neighbors = v[:, :, neighbors, :]  # [B, heads, num_neighbors, dim_head]
            
            # Compute attention scores
            scores = torch.matmul(q_node, k_neighbors.transpose(-2, -1)) * self.scale  # [B, heads, 1, num_neighbors]
            
            # Add relative position encoding if available
            if len(edge_features) > 0:
                # Find edge features for this node's neighbors
                neighbor_edges = []
                for i, neighbor in enumerate(neighbors):
                    if neighbor == node_idx:
                        # Self-connection: zero relative position
                        neighbor_edges.append(torch.zeros(3, device=x.device))
                    else:
                        # Find edge feature (simplified lookup)
                        edge_idx = node_idx * len(neighbors) + i
                        if edge_idx < len(edge_features):
                            neighbor_edges.append(edge_features[edge_idx])
                        else:
                            neighbor_edges.append(torch.zeros(3, device=x.device))
                
                if neighbor_edges:
                    rel_pos = torch.stack(neighbor_edges)  # [num_neighbors, 3]
                    pos_encoding = self.edge_proj(rel_pos).transpose(0, 1)  # [heads, num_neighbors]
                    scores = scores + pos_encoding.unsqueeze(0).unsqueeze(2)  # [B, heads, 1, num_neighbors]
            
            # Apply softmax
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Apply attention to values
            out_node = torch.matmul(attn_weights, v_neighbors)  # [B, heads, 1, dim_head]
            attention_outputs.append(out_node)
        
        # Concatenate all node outputs
        out = torch.cat(attention_outputs, dim=2)  # [B, heads, N, dim_head]
        out = out.transpose(1, 2).contiguous().view(B, N, -1)  # [B, N, heads * dim_head]
        
        return self.to_out(out)


class GraphTransformerBlock(nn.Module):
    """
    Graph transformer block with neighborhood-based attention and feed-forward network.
    """
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.graph_attn = GraphMultiHeadAttention(dim, heads, dim_head, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, edge_index, num_columns):
        # Graph attention with residual connection
        x = x + self.graph_attn(self.norm1(x), edge_index, num_columns)
        
        # Feed-forward with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x


class GraphTransformer3D(BaseRadiationModel):
    """
    Graph Transformer for 3D atmospheric data inspired by GenCast.
    Uses neighborhood-based self-attention on the ICON grid structure.
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
                 process_vertically=True,  # Whether to process vertical interactions
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.device = device
        self.channels_out = channels_out
        self.embed_dim = embed_dim
        self.process_vertically = process_vertically
        
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
        self.pos_embedding = nn.Parameter(torch.randn(1, total_cols * num_height_levels, embed_dim))
        
        # Graph transformer blocks
        mlp_dim = int(embed_dim * mlp_ratio)
        self.transformer_blocks = nn.ModuleList([
            GraphTransformerBlock(embed_dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])
        
        # Vertical processing (if enabled)
        if process_vertically:
            self.vertical_transformer = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=heads,
                    dim_feedforward=mlp_dim,
                    dropout=dropout,
                    batch_first=True
                )
                for _ in range(depth // 2)  # Use half the layers for vertical processing
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
        
        # Flatten for graph processing: [B, N*L, embed_dim]
        x_flat = x.view(B, N * L, -1)
        
        # Add position embeddings
        x_flat = x_flat + self.pos_embedding[:, :N*L, :]
        
        # Apply graph transformer blocks (horizontal processing)
        for block in self.transformer_blocks:
            x_flat = block(x_flat, edge_index, num_columns)
        
        # Reshape back to 3D structure: [B, N, L, embed_dim]
        x = x_flat.view(B, N, L, -1)
        
        # Vertical processing (if enabled)
        if self.process_vertically and hasattr(self, 'vertical_transformer'):
            # Process each column vertically
            x_vert = x.view(B * N, L, -1)  # [B*N, L, embed_dim]
            
            for vert_layer in self.vertical_transformer:
                x_vert = vert_layer(x_vert)
            
            x = x_vert.view(B, N, L, -1)  # [B, N, L, embed_dim]
        
        # Final processing
        x = self.norm(x)
        x = self.output_proj(x)
        x = self.sigmoid(x)
        
        # Scale output
        output = self._scale_output(x, x2d_orig)
        
        return output 