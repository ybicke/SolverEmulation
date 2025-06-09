import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_methods import BaseRadiationModel
from .graph_3d_full import get_3d_graph
import math
from .data_utils import get_triangle_indices


class TraditionalNeighborhoodSelfAttention(nn.Module):
    """
    Traditional graph transformer self-attention that follows the graph structure.
    Each node attends to its k-hop neighbors as defined by the actual graph edges.
    This includes both vertical and horizontal connections as they exist in the graph.
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
        
    def get_k_hop_neighbors(self, edge_index, num_nodes, k_hops):
        """
        Compute k-hop neighborhoods following the actual graph structure.
        This is the traditional graph transformer approach.
        
        Returns a dictionary mapping each node to its k-hop neighbors.
        """
        # Create adjacency list from edge index
        adj_list = {i: set() for i in range(num_nodes)}
        for i in range(edge_index.size(1)):
            src, dst = edge_index[:, i].tolist()
            if 0 <= src < num_nodes and 0 <= dst < num_nodes:
                adj_list[src].add(dst)
        
        # Compute k-hop neighborhoods
        neighborhoods = {}
        for node in range(num_nodes):
            current_hop = {node}  # Start with the node itself
            all_neighbors = {node}
            
            for hop in range(k_hops):
                next_hop = set()
                for n in current_hop:
                    next_hop.update(adj_list[n])
                current_hop = next_hop - all_neighbors  # Only new nodes
                all_neighbors.update(current_hop)
                
                if not current_hop:  # No new neighbors found
                    break
            
            neighborhoods[node] = sorted(list(all_neighbors))
        
        return neighborhoods
    
    def forward(self, x, edge_index, num_columns):
        """
        x: [batch, num_nodes, dim] - where num_nodes is per-batch (N * L)
        edge_index: [2, num_edges] - properly batched edge connectivity
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
        
        # Initialize output
        out = torch.zeros_like(q)
        
        # Process each batch sample separately (they should not interact)
        for batch_idx in range(B):
            # Extract edges for this batch sample
            start_node = batch_idx * N_per_batch
            end_node = (batch_idx + 1) * N_per_batch
            
            # Find edges that belong to this batch sample
            mask = (edge_index[0] >= start_node) & (edge_index[0] < end_node) & \
                   (edge_index[1] >= start_node) & (edge_index[1] < end_node)
            
            if mask.any():
                batch_edge_index = edge_index[:, mask]
                # Adjust edge indices to be relative to this batch (0 to N_per_batch-1)
                batch_edge_index = batch_edge_index - start_node
            else:
                # No edges for this batch, create empty edge index
                batch_edge_index = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
            
            # Get traditional k-hop neighborhoods using the batch-specific edge index
            neighborhoods = self.get_k_hop_neighbors(batch_edge_index, N_per_batch, self.max_hops)
            
            # Debug info (only for first batch to avoid spam)
            if batch_idx == 0:
                neighborhood_sizes = [len(neighbors) for neighbors in neighborhoods.values()]
                if len(neighborhood_sizes) > 0:
                    avg_neighborhood_size = sum(neighborhood_sizes) / len(neighborhood_sizes)
                    max_neighborhood_size = max(neighborhood_sizes)
                    min_neighborhood_size = min(neighborhood_sizes)
                    print(f"Batch {batch_idx} - Traditional Graph Neighborhood Stats:")
                    print(f"  Avg: {avg_neighborhood_size:.1f}, Min: {min_neighborhood_size}, Max: {max_neighborhood_size}, Nodes: {N_per_batch}")
                    print(f"  Memory complexity per batch: O({N_per_batch} × {max_neighborhood_size}) = {N_per_batch*max_neighborhood_size}")
                    
                    # Analyze traditional neighborhood structure for sample nodes
                    sample_nodes = [0, N_per_batch//4, N_per_batch//2]
                    print(f"  Sample traditional neighborhood analysis:")
                    for node_id in sample_nodes:
                        if node_id in neighborhoods:
                            col_id = node_id // num_height_levels
                            level_id = node_id % num_height_levels
                            neighbors = neighborhoods[node_id]
                            
                            # Count vertical neighbors (same column)
                            vertical_neighbors = [n for n in neighbors if n // num_height_levels == col_id]
                            
                            # Count horizontal neighbors (different columns)
                            horizontal_neighbors = [n for n in neighbors if n // num_height_levels != col_id]
                            
                            # Count neighbors by height level
                            neighbor_levels = {}
                            for n in neighbors:
                                n_level = n % num_height_levels
                                neighbor_levels[n_level] = neighbor_levels.get(n_level, 0) + 1
                            
                            print(f"    Node {node_id} (col {col_id}, level {level_id}):")
                            print(f"      Total neighbors: {len(neighbors)}")
                            print(f"      Vertical (same col): {len(vertical_neighbors)}")
                            print(f"      Horizontal (diff cols): {len(horizontal_neighbors)}")
                            print(f"      Neighbors at same level: {neighbor_levels.get(level_id, 0)}")
            
            # Process attention for each node in this batch using only its neighborhood
            for node_idx in range(N_per_batch):
                neighbors = neighborhoods.get(node_idx, [node_idx])
                
                if not neighbors:
                    neighbors = [node_idx]  # Fallback to self-attention
                
                # Extract features for this node and its neighbors within this batch
                q_node = q[batch_idx, node_idx:node_idx+1, :, :]  # [1, heads, dim_head]
                k_neighbors = k[batch_idx, neighbors, :, :]  # [num_neighbors, heads, dim_head]
                v_neighbors = v[batch_idx, neighbors, :, :]  # [num_neighbors, heads, dim_head]
                
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
                    out[batch_idx, node_idx, head, :] = torch.matmul(attn_weights, v_h)
        
        # Reshape output
        out = out.view(B, N_per_batch, -1)  # [B, N_per_batch, heads * dim_head]
        return self.to_out(out)


class TraditionalGraphTransformerLayer(nn.Module):
    """
    Traditional Graph Transformer layer with neighborhood attention and feed-forward.
    """
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0, max_hops=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TraditionalNeighborhoodSelfAttention(dim, heads, dim_head, dropout, max_hops)
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


class TraditionalGraphTransformer3D(BaseRadiationModel):
    """
    Traditional Graph Transformer for 3D atmospheric data on ICON grid.
    
    Key features:
    1. Follows the actual graph structure (both vertical and horizontal edges)
    2. k-hop neighborhood expansion along graph connectivity
    3. Standard graph transformer approach
    4. Direct comparison to hybrid approach
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
                 max_hops=2,  # Maximum hops for neighborhood attention
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.device = device
        self.channels_out = channels_out
        self.embed_dim = embed_dim
        self.num_height_levels = num_height_levels
        
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
        
        # Graph Transformer layers
        mlp_dim = int(embed_dim * mlp_ratio)
        self.graph_layers = nn.ModuleList([
            TraditionalGraphTransformerLayer(embed_dim, heads, dim_head, mlp_dim, dropout, max_hops)
            for _ in range(depth)
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
        print(f"Traditional Graph Transformer Input shape: B={B}, N={N}, L={L}")
        print(f"Edge index shape: {edge_index.shape}")
        print(f"Edge index min/max: {edge_index.min().item()}/{edge_index.max().item()}")
        print(f"Num columns from graph: {num_columns}")
        print(f"Nodes per batch: {N * L}")
        print(f"Using traditional k-hop graph expansion")
        print(f"Position embedding shape: {self.pos_embedding_3d.shape}")
        
        # Prepare features
        x2d_repeated = x2d_norm.unsqueeze(2).repeat(1, 1, L, 1)
        x_concat = torch.cat([x3d_norm, x2d_repeated], dim=-1)
        
        # Project to embedding space
        x = self.input_proj(x_concat)  # [B, N, L, embed_dim]
        
        # Flatten for graph processing
        x_flat = x.view(B, N * L, -1)  # [B, N*L, embed_dim]
        
        # Add position embeddings - ensure sizes match
        pos_emb_size = min(x_flat.size(1), self.pos_embedding_3d.size(1))
        x_flat[:, :pos_emb_size, :] = x_flat[:, :pos_emb_size, :] + self.pos_embedding_3d[:, :pos_emb_size, :]
        
        # Traditional graph transformer processing
        # Follows actual graph structure with k-hop expansion
        for layer in self.graph_layers:
            x_flat = layer(x_flat, edge_index, num_columns)
        
        # Reshape back to 3D structure
        x = x_flat.view(B, N, L, -1)  # [B, N, L, embed_dim]
        
        # Final processing
        x = self.norm(x)
        x = self.output_proj(x)
        x = self.sigmoid(x)
        
        # Scale output
        output = self._scale_output(x, x2d_orig)
        
        return output 