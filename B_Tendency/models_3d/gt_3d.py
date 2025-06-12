import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.graph_3d import get_3d_graph
from utils.data_utils import get_triangle_indices


class NeighborhoodSelfAttention(nn.Module):
    """
    Graph transformer self-attention that follows the graph structure.
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
        
    def get_k_hop_neighbors(self, edge_index, num_nodes, k_hops, cached_neighborhoods=None):
        """
        Compute k-hop neighborhoods following the actual graph structure.
        This is the graph transformer approach.
        
        Returns a dictionary mapping each node to its k-hop neighbors.
        """
        # Build cache if not exists (graph structure never changes)
        if cached_neighborhoods is None:
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
            
            # Cache the result permanently
            cached_neighborhoods = neighborhoods
        
        return cached_neighborhoods
    
    
    def build_padded_neighborhoods(self, neighborhoods, B, N_per_batch, max_neighbors):
        """
        Build padded neighborhood tensors directly from neighborhoods dict.
        Since graph structure is identical across batches, we create it once and broadcast.
        """
        # Build single-batch structure
        single_batch_neighbors = torch.zeros(N_per_batch, max_neighbors, dtype=torch.long)
        single_batch_mask = torch.zeros(N_per_batch, max_neighbors, dtype=torch.bool)
        
        for node_idx in range(N_per_batch):
            neighbors = neighborhoods.get(node_idx, [node_idx])
            if not neighbors:
                neighbors = [node_idx]  # Fallback to self-attention
                
            num_neighbors = len(neighbors)
            
            # Fill padded tensor
            single_batch_neighbors[node_idx, :num_neighbors] = torch.tensor(neighbors)
            single_batch_mask[node_idx, :num_neighbors] = True
            
            # Pad with self-indices (safe fallback)
            if num_neighbors < max_neighbors:
                single_batch_neighbors[node_idx, num_neighbors:] = node_idx
        
        # Expand to all batches using broadcasting (much more efficient)
        padded_neighbors = single_batch_neighbors.unsqueeze(0).expand(B, -1, -1).clone()
        neighbor_mask = single_batch_mask.unsqueeze(0).expand(B, -1, -1).clone()
        
        return padded_neighbors, neighbor_mask
    
    def forward(self, x, edge_index, num_columns, shared_cache=None):
        """
        x: [batch, num_nodes, dim] - where num_nodes is per-batch (N * L)
        edge_index: [2, num_edges] - properly batched edge connectivity
        shared_cache: Dictionary with shared caching structures from the model
        """
        B, N_per_batch, D = x.shape
        device = x.device
    
        
        # Use shared cache if provided, otherwise create local cache
        if shared_cache is not None:
            # Use model-level shared cache
            if shared_cache['neighborhoods'] is None:
                # Build neighborhood cache once (graph structure never changes) - EXPENSIVE!
                single_batch_edges = edge_index[:, :edge_index.shape[1] // B] if B > 1 else edge_index
                neighborhoods = self.get_k_hop_neighbors(
                    single_batch_edges, N_per_batch, self.max_hops, 
                    shared_cache['neighborhoods']
                )
                shared_cache['neighborhoods'] = neighborhoods
            
            # Build padded neighborhoods once (can be cached) - MODERATELY EXPENSIVE!
            if shared_cache['padded_neighbors'] is None:
                neighborhoods = shared_cache['neighborhoods']
                # Compute max_neighbors on the fly (cheap operation)
                max_neighbors = max(len(neighbors) for neighbors in neighborhoods.values())
                
                padded_neighbors, neighbor_mask = self.build_padded_neighborhoods(
                    neighborhoods, B, N_per_batch, max_neighbors
                )
                shared_cache['padded_neighbors'] = padded_neighbors.to(device)
                shared_cache['neighbor_mask'] = neighbor_mask.to(device)
                shared_cache['max_neighbors'] = max_neighbors  # Store for convenience
            
            # Get cached padded structures from shared cache
            padded_neighbors = shared_cache['padded_neighbors']
            neighbor_mask = shared_cache['neighbor_mask']
            max_neighbors = shared_cache['max_neighbors']
        else:
            # Fallback to local computation (should not happen in normal usage)
            raise ValueError("shared_cache must be provided")
        
        # Project to Q, K, V
        q = self.to_q(x)  # [B, N_per_batch, heads * dim_head]
        k = self.to_k(x)  # [B, N_per_batch, heads * dim_head]
        v = self.to_v(x)  # [B, N_per_batch, heads * dim_head]
        
        # Reshape for multi-head attention
        q = q.view(B, N_per_batch, self.heads, self.dim_head)  # [B, N_per_batch, heads, dim_head]
        k = k.view(B, N_per_batch, self.heads, self.dim_head)  # [B, N_per_batch, heads, dim_head]
        v = v.view(B, N_per_batch, self.heads, self.dim_head)  # [B, N_per_batch, heads, dim_head]
        
        # === VECTORIZED ATTENTION COMPUTATION - NO LOOPS! ===
        
        # Expand padded_neighbors for all heads: [B, N_per_batch, heads, max_neighbors]
        neighbor_indices = padded_neighbors.unsqueeze(2).expand(-1, -1, self.heads, -1)
        
        # Gather K and V for neighbors using advanced indexing
        # First, create indices for batch and head dimensions
        batch_indices = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, N_per_batch, self.heads, max_neighbors)
        head_indices = torch.arange(self.heads, device=device).view(1, 1, self.heads, 1).expand(B, N_per_batch, self.heads, max_neighbors)
        
        # Gather neighbor features: [B, N_per_batch, heads, max_neighbors, dim_head]
        k_neighbors = k[batch_indices, neighbor_indices, head_indices]
        v_neighbors = v[batch_indices, neighbor_indices, head_indices]
        
        # Compute attention scores - FULLY VECTORIZED!
        q_expanded = q.unsqueeze(3)  # [B, N_per_batch, heads, 1, dim_head]
        scores = torch.matmul(q_expanded, k_neighbors.transpose(-2, -1)) * self.scale
        # Result: [B, N_per_batch, heads, 1, max_neighbors]
        scores = scores.squeeze(3)  # [B, N_per_batch, heads, max_neighbors]
        
        # Apply neighbor mask
        mask_value = -1e9
        neighbor_mask_expanded = neighbor_mask.unsqueeze(2)  # [B, N_per_batch, 1, max_neighbors]
        scores = scores.masked_fill(~neighbor_mask_expanded, mask_value)
        
        # Apply softmax and dropout - FULLY VECTORIZED!
        attn_weights = F.softmax(scores, dim=-1)  # [B, N_per_batch, heads, max_neighbors]
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values - FULLY VECTORIZED!
        attn_weights_expanded = attn_weights.unsqueeze(-1)  # [B, N_per_batch, heads, max_neighbors, 1]
        out = torch.sum(attn_weights_expanded * v_neighbors, dim=3)  # [B, N_per_batch, heads, dim_head]
        
        # Reshape back to original format
        out = out.view(B, N_per_batch, -1)  # [B, N_per_batch, heads * dim_head]
        
        return self.to_out(out)


class GraphTransformerLayer(nn.Module):
    """
    Traditional Graph Transformer layer with neighborhood attention and feed-forward.
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
        
    def forward(self, x, edge_index, num_columns, shared_cache=None):
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x), edge_index, num_columns, shared_cache)
        
        # Feed-forward with residual connection  
        x = x + self.mlp(self.norm2(x))
        
        return x


class GraphTransformer3D(nn.Module):
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
            GraphTransformerLayer(embed_dim, heads, dim_head, mlp_dim, dropout, max_hops)
            for _ in range(depth)
        ])
        
        # Output processing
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, channels_out)
        self.sigmoid = nn.Sigmoid()
        
        # Caching for neighborhood computation (only cache expensive operations)
        self._cached_neighborhoods = None      # EXPENSIVE: k-hop graph traversal
        self._cached_padded_neighbors = None   # MODERATELY EXPENSIVE: tensor creation/padding
        self._cached_neighbor_mask = None      # MODERATELY EXPENSIVE: boolean tensor
        self._max_neighbors = None             # CHEAP: but stored for convenience
        
    def _get_shared_cache(self):
        """Get shared cache dictionary for all attention layers."""
        return {
            'neighborhoods': self._cached_neighborhoods,
            'padded_neighbors': self._cached_padded_neighbors,
            'neighbor_mask': self._cached_neighbor_mask,
            'max_neighbors': self._max_neighbors
        }
        
    def forward(self, x3d_norm, x2d_norm):
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
        
        # Traditional graph transformer processing
        shared_cache = self._get_shared_cache()
        for layer in self.graph_layers:
            x_flat = layer(x_flat, edge_index, num_columns, shared_cache)
        
        # Reshape back to 3D structure
        x = x_flat.view(B, N, L, -1)  # [B, N, L, embed_dim]
        
        # Final processing
        x = self.norm(x)
        x = self.output_proj(x)
        
        
        return x 