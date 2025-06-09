import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_methods import BaseRadiationModel
from .graph_3d_full import get_3d_graph
import math
from .data_utils import get_triangle_indices


class OptimizedNeighborhoodSelfAttention(nn.Module):
    """
    Optimized graph transformer self-attention with:
    1. Cached neighborhood structures
    2. Vectorized attention computation
    3. Sparse attention masks
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
        
        # Cached structures
        self._cached_neighborhoods = None
        self._cached_attention_mask = None
        self._cached_edge_config = None
        
    def _build_neighborhood_cache(self, edge_index, num_nodes, k_hops):
        """
        Build and cache neighborhood structures for efficient attention.
        Returns attention mask and index mappings.
        """
        # Create adjacency list from edge index
        adj_list = {i: set() for i in range(num_nodes)}
        for i in range(edge_index.size(1)):
            src, dst = edge_index[:, i].tolist()
            if 0 <= src < num_nodes and 0 <= dst < num_nodes:
                adj_list[src].add(dst)
        
        # Compute k-hop neighborhoods
        neighborhoods = {}
        max_neighbors = 0
        
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
            max_neighbors = max(max_neighbors, len(neighborhoods[node]))
        
        # Create dense attention mask and index mapping
        # attention_mask[i, j] = 1 if node j is in node i's neighborhood
        attention_mask = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)
        
        # Also create a more efficient sparse representation
        node_indices = []
        neighbor_indices = []
        
        for node_idx in range(num_nodes):
            neighbors = neighborhoods.get(node_idx, [node_idx])
            for neighbor_idx in neighbors:
                attention_mask[node_idx, neighbor_idx] = True
                node_indices.append(node_idx)
                neighbor_indices.append(neighbor_idx)
        
        # Convert to tensors
        sparse_indices = torch.stack([
            torch.tensor(node_indices, dtype=torch.long),
            torch.tensor(neighbor_indices, dtype=torch.long)
        ])
        
        return {
            'neighborhoods': neighborhoods,
            'attention_mask': attention_mask,
            'sparse_indices': sparse_indices,
            'max_neighbors': max_neighbors,
            'num_nodes': num_nodes
        }
    
    def forward(self, x, edge_index, num_columns):
        """
        x: [batch, num_nodes, dim] - where num_nodes is per-batch (N * L)
        edge_index: [2, num_edges] - properly batched edge connectivity
        """
        B, N_per_batch, D = x.shape
        device = x.device
        
        # Calculate num_height_levels from the data dimensions
        num_height_levels = N_per_batch // num_columns
        
        # Create cache key for this configuration
        edge_config = (N_per_batch, edge_index.shape[1], self.max_hops)
        
        # Check if we need to rebuild cache
        if (self._cached_neighborhoods is None or 
            self._cached_edge_config != edge_config):
            
            print(f"Building neighborhood cache for {N_per_batch} nodes, {self.max_hops} hops...")
            
            # Extract edges for single batch (assumes all batches have same structure)
            start_node = 0
            end_node = N_per_batch
            
            # Find edges that belong to first batch sample
            mask = (edge_index[0] >= start_node) & (edge_index[0] < end_node) & \
                   (edge_index[1] >= start_node) & (edge_index[1] < end_node)
            
            if mask.any():
                batch_edge_index = edge_index[:, mask]
                # Adjust edge indices to be relative to single batch
                batch_edge_index = batch_edge_index % N_per_batch
            else:
                batch_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            
            # Build cache
            self._cached_neighborhoods = self._build_neighborhood_cache(
                batch_edge_index, N_per_batch, self.max_hops
            )
            self._cached_edge_config = edge_config
            
            # Move cache to correct device
            self._cached_neighborhoods['attention_mask'] = self._cached_neighborhoods['attention_mask'].to(device)
            self._cached_neighborhoods['sparse_indices'] = self._cached_neighborhoods['sparse_indices'].to(device)
            
            # Print cache statistics
            neighborhoods = self._cached_neighborhoods['neighborhoods']
            neighborhood_sizes = [len(neighbors) for neighbors in neighborhoods.values()]
            avg_size = sum(neighborhood_sizes) / len(neighborhood_sizes)
            max_size = max(neighborhood_sizes)
            min_size = min(neighborhood_sizes)
            
            print(f"Cached neighborhood stats:")
            print(f"  Nodes: {N_per_batch}, Avg neighbors: {avg_size:.1f}")
            print(f"  Min: {min_size}, Max: {max_size}")
            print(f"  Memory savings: Cached once vs recomputed {B}×{len(self.to_out)} times per forward")
        
        # Get cached structures
        attention_mask = self._cached_neighborhoods['attention_mask']  # [N_per_batch, N_per_batch]
        
        # Project to Q, K, V
        q = self.to_q(x)  # [B, N_per_batch, heads * dim_head]
        k = self.to_k(x)  # [B, N_per_batch, heads * dim_head]
        v = self.to_v(x)  # [B, N_per_batch, heads * dim_head]
        
        # Reshape for multi-head attention
        q = q.view(B, N_per_batch, self.heads, self.dim_head).transpose(1, 2)  # [B, heads, N_per_batch, dim_head]
        k = k.view(B, N_per_batch, self.heads, self.dim_head).transpose(1, 2)  # [B, heads, N_per_batch, dim_head]
        v = v.view(B, N_per_batch, self.heads, self.dim_head).transpose(1, 2)  # [B, heads, N_per_batch, dim_head]
        
        # Vectorized attention computation
        # Compute attention scores: [B, heads, N_per_batch, N_per_batch]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply neighborhood mask (broadcast across batch and heads)
        # Mask out attention to non-neighbors with large negative value
        mask_value = -1e9
        attention_mask_expanded = attention_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, N_per_batch, N_per_batch]
        scores = scores.masked_fill(~attention_mask_expanded, mask_value)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values: [B, heads, N_per_batch, dim_head]
        out = torch.matmul(attn_weights, v)
        
        # Reshape back: [B, N_per_batch, heads * dim_head]
        out = out.transpose(1, 2).contiguous().view(B, N_per_batch, -1)
        
        return self.to_out(out)


class OptimizedGraphTransformerLayer(nn.Module):
    """
    Optimized Graph Transformer layer with cached neighborhood attention.
    """
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0, max_hops=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = OptimizedNeighborhoodSelfAttention(dim, heads, dim_head, dropout, max_hops)
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


class OptimizedGraphTransformer3D(BaseRadiationModel):
    """
    Optimized Graph Transformer for 3D atmospheric data on ICON grid.
    
    Key optimizations:
    1. Cached neighborhood structures (computed once, reused everywhere)
    2. Vectorized attention computation (no nested loops)
    3. Sparse attention masks for memory efficiency
    4. GPU-optimized tensor operations
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
                 max_hops=2,
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
        
        # Learnable position embeddings
        self.pos_embedding_3d = nn.Parameter(torch.randn(1, actual_num_columns * num_height_levels, embed_dim))
        
        # Optimized Graph Transformer layers
        mlp_dim = int(embed_dim * mlp_ratio)
        self.graph_layers = nn.ModuleList([
            OptimizedGraphTransformerLayer(embed_dim, heads, dim_head, mlp_dim, dropout, max_hops)
            for _ in range(depth)
        ])
        
        # Output processing
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, channels_out)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x3d_norm, x2d_norm, x2d_orig):
        B, N, L, _ = x3d_norm.shape
        
        # Get graph structure (this could also be cached if graph is constant)
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
        print(f"Optimized Graph Transformer Input shape: B={B}, N={N}, L={L}")
        print(f"Edge index shape: {edge_index.shape}")
        print(f"Num columns from graph: {num_columns}")
        print(f"Nodes per batch: {N * L}")
        print(f"Using optimized k-hop graph expansion with caching")
        
        # Prepare features
        x2d_repeated = x2d_norm.unsqueeze(2).repeat(1, 1, L, 1)
        x_concat = torch.cat([x3d_norm, x2d_repeated], dim=-1)
        
        # Project to embedding space
        x = self.input_proj(x_concat)  # [B, N, L, embed_dim]
        
        # Flatten for graph processing
        x_flat = x.view(B, N * L, -1)  # [B, N*L, embed_dim]
        
        # Add position embeddings
        pos_emb_size = min(x_flat.size(1), self.pos_embedding_3d.size(1))
        x_flat[:, :pos_emb_size, :] = x_flat[:, :pos_emb_size, :] + self.pos_embedding_3d[:, :pos_emb_size, :]
        
        # Optimized graph transformer processing
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