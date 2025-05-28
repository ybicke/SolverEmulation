import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_methods import BaseRadiationModel
from .graph_3d_full import get_3d_graph
import math
from .data_utils import get_triangle_indices


class NeighborhoodSelfAttention(nn.Module):
    """
    GenCast-style neighborhood-based self-attention.
    Each node only attends to its k-hop neighbors as defined by the mesh connectivity.
    This dramatically reduces memory complexity from O(n²) to O(n × k).
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, max_hops=2, use_edge_features=True):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.max_hops = max_hops
        self.use_edge_features = use_edge_features
        inner_dim = dim_head * heads
        
        # Node feature projections
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        
        # Edge feature processing (relative position encoding)
        if use_edge_features:
            self.edge_encoder = nn.Sequential(
                nn.Linear(3, dim),  # 3D relative positions
                nn.ReLU(),
                nn.Linear(dim, heads)
            )
        
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, dim)
        
    def get_k_hop_neighbors(self, edge_index, num_nodes, k_hops, num_columns, num_height_levels):
        """
        Compute hybrid neighborhoods that include:
        1. All vertical levels in the same column (full vertical attention)
        2. k-hop horizontal neighbors at the same height level
        
        Returns a dictionary mapping each node to its hybrid neighbors.
        """
        # Create adjacency list from edge index
        adj_list = {i: set() for i in range(num_nodes)}
        for i in range(edge_index.size(1)):
            src, dst = edge_index[:, i].tolist()
            if 0 <= src < num_nodes and 0 <= dst < num_nodes:
                adj_list[src].add(dst)
        
        # Create hybrid neighborhoods for each node
        neighborhoods = {}
        for node_id in range(num_nodes):
            # Decode node position: col_id, height_level
            col_id = node_id // num_height_levels
            height_level = node_id % num_height_levels
            
            # Start with the node itself
            hybrid_neighbors = {node_id}
            
            # 1. Add ALL vertical levels in the same column
            for h in range(num_height_levels):
                vertical_node = col_id * num_height_levels + h
                if vertical_node < num_nodes:
                    hybrid_neighbors.add(vertical_node)
            
            # 2. Add k-hop horizontal neighbors at the SAME height level
            # Start from current node for horizontal expansion
            current_hop_horizontal = {node_id}
            all_horizontal_neighbors = {node_id}
            
            for hop in range(k_hops):
                next_hop = set()
                for n in current_hop_horizontal:
                    # Only consider neighbors at the same height level
                    n_col = n // num_height_levels
                    n_height = n % num_height_levels
                    
                    if n_height == height_level:  # Same height level
                        for neighbor in adj_list[n]:
                            neighbor_col = neighbor // num_height_levels
                            neighbor_height = neighbor % num_height_levels
                            
                            # Only add if it's at the same height level (horizontal neighbor)
                            if neighbor_height == height_level and neighbor_col != col_id:
                                next_hop.add(neighbor)
                
                # Only add new horizontal neighbors (not already found)
                current_hop_horizontal = next_hop - all_horizontal_neighbors
                all_horizontal_neighbors.update(current_hop_horizontal)
                
                if not current_hop_horizontal:  # No new horizontal neighbors found
                    break
            
            # Add horizontal neighbors to hybrid neighborhood
            hybrid_neighbors.update(all_horizontal_neighbors)
            
            # For each horizontal neighbor column, also add ALL vertical levels
            horizontal_columns = set()
            for h_neighbor in all_horizontal_neighbors:
                h_col = h_neighbor // num_height_levels
                horizontal_columns.add(h_col)
            
            # Add all vertical levels from horizontal neighbor columns
            for h_col in horizontal_columns:
                for h in range(num_height_levels):
                    vertical_node = h_col * num_height_levels + h
                    if vertical_node < num_nodes:
                        hybrid_neighbors.add(vertical_node)
            
            neighborhoods[node_id] = sorted(list(hybrid_neighbors))
        
        return neighborhoods
    
    def forward(self, x, edge_index, num_columns):
        """
        x: [batch, num_nodes, dim] - where num_nodes is per-batch (N * L)
        edge_index: [2, num_edges] - properly batched edge connectivity (already offset per batch)
        """
        B, N_per_batch, D = x.shape
        # N_per_batch = N * L (nodes per batch sample)
        
        # Calculate num_height_levels from the data dimensions
        # N_per_batch = num_columns * num_height_levels
        num_height_levels = N_per_batch // num_columns
        
        # Project to Q, K, V
        q = self.to_q(x)  # [B, N_per_batch, heads * dim_head]
        k = self.to_k(x)  # [B, N_per_batch, heads * dim_head]
        v = self.to_v(x)  # [B, N_per_batch, heads * dim_head]
        
        # Reshape for multi-head attention
        q = q.view(B, N_per_batch, self.heads, self.dim_head)
        k = k.view(B, N_per_batch, self.heads, self.dim_head) 
        v = v.view(B, N_per_batch, self.heads, self.dim_head)
        
        # Flatten batch and node dimensions for processing with the batched edge index
        q_flat = q.view(B * N_per_batch, self.heads, self.dim_head)  # [B*N_per_batch, heads, dim_head]
        k_flat = k.view(B * N_per_batch, self.heads, self.dim_head)  # [B*N_per_batch, heads, dim_head]  
        v_flat = v.view(B * N_per_batch, self.heads, self.dim_head)  # [B*N_per_batch, heads, dim_head]
        
        # Get neighborhoods using the full batched edge index (already properly offset)
        # correct batch seperation should be handled in the get_3d_graph function here get nodes for nr of nodes per batch
        total_nodes = B * N_per_batch
        neighborhoods = self.get_k_hop_neighbors(edge_index, total_nodes, self.max_hops, num_columns, num_height_levels)
        
        # Debug info
        if True:  # Show for every forward pass to understand the structure
            neighborhood_sizes = [len(neighbors) for neighbors in neighborhoods.values()]
            if len(neighborhood_sizes) > 0:
                avg_neighborhood_size = sum(neighborhood_sizes) / len(neighborhood_sizes)
                max_neighborhood_size = max(neighborhood_sizes)
                min_neighborhood_size = min(neighborhood_sizes)
                print(f"Full Batched - Hybrid Neighborhood Stats:")
                print(f"  Avg: {avg_neighborhood_size:.1f}, Min: {min_neighborhood_size}, Max: {max_neighborhood_size}, Total nodes: {total_nodes}")
                print(f"  Memory complexity: O({total_nodes} × {max_neighborhood_size}) = {total_nodes*max_neighborhood_size}")
                
                # Analyze neighborhood structure for sample nodes from different batches
                sample_nodes = [0, N_per_batch-1, N_per_batch, total_nodes-1]  # First batch first/last, second batch first/last
                print(f"  Sample hybrid neighborhood analysis (across batches):")
                for node_id in sample_nodes:
                    if node_id in neighborhoods:
                        batch_id = node_id // N_per_batch
                        local_node_id = node_id % N_per_batch
                        col_id = local_node_id // num_height_levels
                        level_id = local_node_id % num_height_levels
                        neighbors = neighborhoods[node_id]
                        
                        # Count vertical neighbors (same column, same batch)
                        vertical_neighbors = []
                        horizontal_columns = set()
                        cross_batch_neighbors = 0
                        
                        for n in neighbors:
                            n_batch = n // N_per_batch
                            n_local = n % N_per_batch
                            n_col = n_local // num_height_levels
                            
                            if n_batch == batch_id and n_col == col_id:
                                vertical_neighbors.append(n)
                            elif n_batch == batch_id and n_col != col_id:
                                horizontal_columns.add(n_col)
                            else:
                                cross_batch_neighbors += 1
                        
                        print(f"    Node {node_id} (batch {batch_id}, col {col_id}, level {level_id}):")
                        print(f"      Total neighbors: {len(neighbors)}")
                        print(f"      Vertical (same col): {len(vertical_neighbors)}")
                        print(f"      Horizontal columns: {len(horizontal_columns)}")
                        print(f"      Cross-batch neighbors: {cross_batch_neighbors}")
        
        # Initialize output
        out_flat = torch.zeros_like(q_flat)
        
        # Process attention for each node using its neighborhood
        for node_idx in range(total_nodes):
            neighbors = neighborhoods.get(node_idx, [node_idx])
            
            if not neighbors:
                neighbors = [node_idx]  # Fallback to self-attention
            
            # Extract features for this node and its neighbors
            q_node = q_flat[node_idx:node_idx+1, :, :]  # [1, heads, dim_head]
            k_neighbors = k_flat[neighbors, :, :]  # [num_neighbors, heads, dim_head]
            v_neighbors = v_flat[neighbors, :, :]  # [num_neighbors, heads, dim_head]
            
            # Compute attention scores for each head
            for head in range(self.heads):
                q_h = q_node[0, head:head+1, :]  # [1, dim_head]
                k_h = k_neighbors[:, head, :]    # [num_neighbors, dim_head]
                v_h = v_neighbors[:, head, :]    # [num_neighbors, dim_head]
                
                # Attention scores: [1, num_neighbors]
                scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * self.scale
                
                # Apply attention weights
                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = self.dropout(attn_weights)
                
                # Compute output: [1, dim_head]
                out_flat[node_idx, head, :] = torch.matmul(attn_weights, v_h)
        
        # Reshape output back to batch format
        out = out_flat.view(B, N_per_batch, self.heads, self.dim_head)
        out = out.view(B, N_per_batch, -1)  # [B, N_per_batch, heads * dim_head]
        return self.to_out(out)


class GraphTransformerLayer(nn.Module):
    """
    GenCast-style Graph Transformer layer with neighborhood attention and feed-forward.
    """
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0, max_hops=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = NeighborhoodSelfAttention(dim, heads, dim_head, dropout, max_hops)
        self.norm2 = nn.LayerNorm(dim)
        
        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, edge_index, num_columns):
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x), edge_index, num_columns)
        
        # Feed-forward with residual connection  
        x = x + self.mlp(self.norm2(x))
        
        return x


class GenCastTransformer3D(BaseRadiationModel):
    """
    GenCast-inspired Graph Transformer for 3D atmospheric data on ICON grid.
    
    Key features:
    1. Neighborhood-based self-attention (only immediate neighbors)
    2. Relative position encoding for spatial relationships
    3. Separate processing for horizontal (graph) and vertical (standard) interactions
    4. Efficient scaling with mesh resolution
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
                 vertical_layers=None,  # Number of vertical processing layers
                 max_hops=2,  # Maximum hops for neighborhood attention
                 horizontal_only_graph=False,  # NEW: Use only horizontal edges in graph attention
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.device = device
        self.channels_out = channels_out
        self.embed_dim = embed_dim
        self.num_height_levels = num_height_levels
        self.horizontal_only_graph = horizontal_only_graph
        
        # Store grid parameters
        self.grid_file_path = grid_file_path
        self.triangle_id = triangle_id
        self.division_factor = division_factor
        self.total_cols = total_cols
        self.fully_connected = fully_connected
        self.disable_horizontal = disable_horizontal
        
        # Input projection
        total_channels = channels_in_3d + channels_in_2d
        self.input_proj = nn.Linear(total_channels, embed_dim)
        
        # Get the actual number of columns we'll be processing
        triangle_indices = get_triangle_indices(triangle_id, division_factor, total_cols)
        actual_num_columns = len(triangle_indices)
        
        # Learnable position embeddings - use actual size, not total_cols
        self.pos_embedding_3d = nn.Parameter(torch.randn(1, actual_num_columns * num_height_levels, embed_dim))
        
        # Horizontal processing: Graph Transformer layers
        mlp_dim = int(embed_dim * mlp_ratio)
        self.horizontal_layers = nn.ModuleList([
            GraphTransformerLayer(embed_dim, heads, dim_head, mlp_dim, dropout, max_hops)
            for _ in range(depth)
        ])
        
        # Vertical processing: Standard transformer layers
        vertical_layers = vertical_layers or max(1, depth // 2)
        self.vertical_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            )
            for _ in range(vertical_layers)
        ])
        
        # Output processing
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
        
        # Debug information
        print(f"Input shape: B={B}, N={N}, L={L}")
        print(f"Edge index shape: {edge_index.shape}")
        print(f"Edge index min/max: {edge_index.min().item()}/{edge_index.max().item()}")
        print(f"Num columns from graph: {num_columns}")
        print(f"Nodes per batch: {N * L}")
        print(f"Processing batches separately (no cross-batch interactions)")
        print(f"Position embedding shape: {self.pos_embedding_3d.shape}")
        
        # Prepare features
        x2d_repeated = x2d_norm.unsqueeze(2).repeat(1, 1, L, 1)
        x_concat = torch.cat([x3d_norm, x2d_repeated], dim=-1)
        
        # Project to embedding space
        x = self.input_proj(x_concat)  # [B, N, L, embed_dim]
        
        # Flatten for graph processing - combine batch and spatial dimensions
        x_flat = x.view(B, N * L, -1)  # [B, N*L, embed_dim]
        
        # Add position embeddings - ensure sizes match
        pos_emb_size = min(x_flat.size(1), self.pos_embedding_3d.size(1))
        x_flat[:, :pos_emb_size, :] = x_flat[:, :pos_emb_size, :] + self.pos_embedding_3d[:, :pos_emb_size, :]
        
        # Horizontal processing with graph transformer
        # Process each batch separately - no cross-batch interactions
        # NOTE: This now includes both horizontal AND vertical attention via hybrid neighborhoods
        for layer in self.horizontal_layers:
            x_flat = layer(x_flat, edge_index, num_columns)
        
        # Reshape back to 3D structure
        x = x_flat.view(B, N, L, -1)  # [B, N, L, embed_dim]
        
        # Optional: Additional vertical processing within each column
        # (May be redundant since hybrid neighborhoods already include full vertical attention)
        if len(self.vertical_layers) > 0:
            x_vert = x.view(B * N, L, -1)  # [B*N, L, embed_dim]
            
            for layer in self.vertical_layers:
                x_vert = layer(x_vert)
            
            x = x_vert.view(B, N, L, -1)  # [B, N, L, embed_dim]
        
        # Final processing
        x = self.norm(x)
        x = self.output_proj(x)
        x = self.sigmoid(x)
        
        # Scale output
        output = self._scale_output(x, x2d_orig)
        
        return output 