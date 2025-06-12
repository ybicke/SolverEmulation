import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.graph_3d import get_3d_graph
from utils.data_utils import get_triangle_indices


class LargeNeighborhoodAttention(nn.Module):
    """
    Hybrid attention mechanism that combines:
    1. Full vertical attention within each column (atmospheric physics)
    2. k-hop horizontal attention respecting spherical geometry
    
    This approach is physically motivated for atmospheric modeling.
    """
    def __init__(self, dim, heads, dim_head, dropout, max_hops):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.max_hops = max_hops
        inner_dim = dim_head * heads
        
        # Node feature projections
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, dim)
      
    def get_large_neighborhoods(self, edge_index, num_nodes, num_columns, num_height_levels, cached_neighborhoods=None):
        """
        Compute hybrid neighborhoods efficiently.
        
        For each node, the neighborhood includes:
        1. All nodes in the same column (full vertical attention)
        2. All nodes in k-hop neighboring columns (horizontal + their vertical extent)
        
        Returns: dict mapping node_id -> list of neighbor node_ids
        """
        if cached_neighborhoods is not None:
            return cached_neighborhoods
            
        # Build adjacency list from edge index (only need horizontal connectivity)
        adj_list = {i: set() for i in range(num_columns)}
        
        # Extract horizontal connections between columns
        for i in range(edge_index.size(1)): # Loop through all 557,952 edges
            src, dst = edge_index[:, i].tolist()
            if 0 <= src < num_nodes and 0 <= dst < num_nodes:
                src_col = src // num_height_levels # Which column is source in
                dst_col = dst // num_height_levels # Which column is destination in
                
                # Only store column-to-column connections
                if src_col != dst_col: # Searches for horizontal edges
                    adj_list[src_col].add(dst_col)
        
        # For each column, find k-hop neighboring columns
        column_neighborhoods = {}
        for col_id in range(num_columns): # Loop through all 1024 columns
            current_hop = {col_id}
            all_neighbor_cols = {col_id}
            
            for hop in range(self.max_hops):
                next_hop = set()
                for col in current_hop:
                    next_hop.update(adj_list[col])
                current_hop = next_hop - all_neighbor_cols
                all_neighbor_cols.update(current_hop)
                
                if not current_hop:
                    break
            
            column_neighborhoods[col_id] = sorted(list(all_neighbor_cols))
        
        # Create node-level neighborhoods from column neighborhoods
        node_neighborhoods = {}
        for node_id in range(num_nodes):
            col_id = node_id // num_height_levels
            neighbor_cols = column_neighborhoods[col_id]
            
            # All nodes in neighboring columns (includes full vertical extent)
            neighbors = set()
            for neighbor_col in neighbor_cols:
                for h in range(num_height_levels):
                    neighbor_node = neighbor_col * num_height_levels + h
                    if neighbor_node < num_nodes:
                        neighbors.add(neighbor_node)
            
            node_neighborhoods[node_id] = sorted(list(neighbors))
        
        return node_neighborhoods
    
    def build_attention_mask(self, neighborhoods, B, N_per_batch, device):
        """
        Build efficient attention mask for vectorized computation.
        Returns padded neighbor indices and mask.
        - Neighborhoods have different sizes. For vectorized attention, we need fixed-size tensors.
        - padded_neighbors: [71680, 280]
        - node_neighborhoods[0]: 140 neighbors
        - node_neighborhoods[71]: 210 neighbors
        - padded_neighbors[0] = [n0, n1, n2, ..., n139, 0, 0, 0, ..., 0]     # 140 real + 140 padding
        - neighbor_mask[0]    = [T,  T,  T,  ..., T,    F, F, F, ..., F]     # 140 True + 140 False
        """
        # Find maximum neighborhood size
        max_neighbors = max(len(neighbors) for neighbors in neighborhoods.values())
        
        # Build padded neighbor tensor
        padded_neighbors = torch.zeros(N_per_batch, max_neighbors, dtype=torch.long, device=device)
        neighbor_mask = torch.zeros(N_per_batch, max_neighbors, dtype=torch.bool, device=device)
        
        for node_idx in range(N_per_batch):
            neighbors = neighborhoods.get(node_idx, [node_idx])
            if not neighbors:
                neighbors = [node_idx]
                
            num_neighbors = len(neighbors)
            
            # Fill neighbor indices
            padded_neighbors[node_idx, :num_neighbors] = torch.tensor(neighbors, device=device)
            neighbor_mask[node_idx, :num_neighbors] = True
            
            # Pad remaining positions with self-reference (safe fallback)
            if num_neighbors < max_neighbors:
                padded_neighbors[node_idx, num_neighbors:] = node_idx
        
        # Expand for batch dimension
        padded_neighbors = padded_neighbors.unsqueeze(0).expand(B, -1, -1).contiguous()
        neighbor_mask = neighbor_mask.unsqueeze(0).expand(B, -1, -1).contiguous()
        
        return padded_neighbors, neighbor_mask, max_neighbors
    
    def forward(self, x, edge_index, num_columns, shared_cache=None):
        """
        Forward pass with efficient vectorized attention.
        
        Args:
            x: [batch, num_nodes, dim] node features
            edge_index: [2, num_edges] edge connectivity
            num_columns: number of columns in the grid
            shared_cache: shared caching for efficiency
        """
        B, N_per_batch, D = x.shape
        device = x.device
        num_height_levels = N_per_batch // num_columns
        
        # Use shared cache for neighborhoods (expensive computation)
        if shared_cache is not None:
            if shared_cache['neighborhoods'] is None:
                # Only compute on single batch edge index
                single_batch_edges = edge_index[:, :edge_index.shape[1] // B] if B > 1 else edge_index
                neighborhoods = self.get_large_neighborhoods(
                    single_batch_edges, N_per_batch, num_columns, num_height_levels
                )
                shared_cache['neighborhoods'] = neighborhoods
            
            if shared_cache['attention_mask'] is None:
                neighborhoods = shared_cache['neighborhoods']
                padded_neighbors, neighbor_mask, max_neighbors = self.build_attention_mask(
                    neighborhoods, B, N_per_batch, device
                )
                shared_cache['attention_mask'] = (padded_neighbors, neighbor_mask, max_neighbors)
            
            padded_neighbors, neighbor_mask, max_neighbors = shared_cache['attention_mask']
        else:
            raise ValueError("shared_cache must be provided for efficiency")
        
        # Project to Q, K, V
        q = self.to_q(x).view(B, N_per_batch, self.heads, self.dim_head) # eg. [1, 71680, 8, 8]
        k = self.to_k(x).view(B, N_per_batch, self.heads, self.dim_head)
        v = self.to_v(x).view(B, N_per_batch, self.heads, self.dim_head)
        
        # Vectorized neighbor gathering, eg. all shape # [1, 71680, 8, 280]
        neighbor_indices = padded_neighbors.unsqueeze(2).expand(-1, -1, self.heads, -1) # which neighbor to look at
        batch_indices = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, N_per_batch, self.heads, max_neighbors) # which batch to look at, eg all 0 for bs 1
        head_indices = torch.arange(self.heads, device=device).view(1, 1, self.heads, 1).expand(B, N_per_batch, self.heads, max_neighbors) # which head to look at
        
        # Gather neighbor K, V
        k_neighbors = k[batch_indices, neighbor_indices, head_indices] # eg. [1, 71680, 8, 280]
        v_neighbors = v[batch_indices, neighbor_indices, head_indices] 
        
        # Compute attention scores
        q_expanded = q.unsqueeze(3)  # [B, N, heads, 1, dim_head]
        scores = torch.matmul(q_expanded, k_neighbors.transpose(-2, -1)) * self.scale
        scores = scores.squeeze(3)  # [B, N, heads, max_neighbors]
        
        # Apply mask
        mask_value = -1e9
        neighbor_mask_expanded = neighbor_mask.unsqueeze(2)
        scores = scores.masked_fill(~neighbor_mask_expanded, mask_value)
        
        # Attention weights and output
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_weights_expanded = attn_weights.unsqueeze(-1)
        out = torch.sum(attn_weights_expanded * v_neighbors, dim=3)
        
        # Reshape and project output
        out = out.view(B, N_per_batch, -1)
        return self.to_out(out)


class LargeTransformerLayer(nn.Module):
    """
    Transformer layer with hybrid attention and feed-forward network.
    Uses pre-normalization for better gradient flow.
    """
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout, max_hops):
        super().__init__()
        # Pre-normalization layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Hybrid attention
        self.attn = LargeNeighborhoodAttention(dim, heads, dim_head, dropout, max_hops)
        
        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
    
    def forward(self, x, edge_index, num_columns, shared_cache=None):
        # Pre-norm + attention + residual
        x = x + self.attn(self.norm1(x), edge_index, num_columns, shared_cache)
        
        # Pre-norm + MLP + residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class HybridGraphTransformer3D(nn.Module):
    """
    Hybrid Graph Transformer for 3D atmospheric data on ICON grid.
    
    Key features:
    1. Physically-motivated hybrid attention (full vertical + k-hop horizontal)
    2. Efficient vectorized implementation
    3. Proper caching for repeated computations
    4. Training pipeline compatibility
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
                 heads,
                 dim_head,
                 mlp_ratio,
                 fully_connected,
                 disable_horizontal,
                 max_hops,
                 *args,
                 **kwargs):
        super().__init__()
        
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
        self.max_hops = max_hops
        
        # Input projection
        total_channels = channels_in_3d + channels_in_2d
        self.input_proj = nn.Linear(total_channels, embed_dim)
        
        # Get actual number of columns
        triangle_indices = get_triangle_indices(triangle_id, division_factor, total_cols)
        actual_num_columns = len(triangle_indices)
        
        # Position embeddings with proper initialization
        self.pos_embedding_3d = nn.Parameter(
            torch.randn(1, actual_num_columns * num_height_levels, embed_dim) * 0.02
        )
        
        # Hybrid transformer layers
        mlp_dim = int(embed_dim * mlp_ratio)
        self.layers = nn.ModuleList([
            LargeTransformerLayer(embed_dim, heads, dim_head, mlp_dim, dropout, max_hops)
            for _ in range(depth)
        ])
        
        # Output processing
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, channels_out)
        
       
        # Shared cache for efficiency
        self._shared_cache = {
            'neighborhoods': None,
            'attention_mask': None
        }
   

    
    def forward(self, x3d_norm, x2d_norm):
        """
        Forward pass compatible with training pipeline.
        
        Args:
            x3d_norm: [batch, num_columns, num_levels, channels_3d] normalized 3D data
            x2d_norm: [batch, num_columns, channels_2d] normalized 2D data
        
        Returns:
            output: [batch, num_columns, num_levels, channels_out] predictions
        """
        B, N, L, _ = x3d_norm.shape
        
        # Get graph structure (cached)
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
        x_flat[:, :pos_emb_size, :] += self.pos_embedding_3d[:, :pos_emb_size, :]
        
        # Process through hybrid transformer layers
        for layer in self.layers:
            x_flat = layer(x_flat, edge_index, num_columns, self._shared_cache)
        
        # Reshape back to 3D structure
        x = x_flat.view(B, N, L, -1)  # [B, N, L, embed_dim]
        
        # Final processing
        x = self.norm(x)
        x = self.output_proj(x)
        
        return x 