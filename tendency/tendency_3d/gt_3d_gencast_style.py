import torch
import torch.nn as nn
import torch.nn.functional as F
from graph_3d_full import get_3d_graph
from data_utils import get_triangle_indices


class GenCastStyleAttention(nn.Module):
    """
    GenCast-style attention mechanism for ICON triangular grid.
    
    Key features:
    1. Pure 2D attention on triangular grid (no explicit vertical dimension)
    2. All height levels are encoded in node features
    3. k-hop neighborhood attention respecting triangular connectivity
    4. Similar to GenCast but on triangular instead of icosahedral geometry
    
    This approach treats each atmospheric column as a single node with
    rich feature representation encompassing all vertical levels.
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
        

    
    def get_2d_neighborhoods(self, edge_index, num_columns, num_height_levels, cached_neighborhoods=None):
        """
        Compute k-hop neighborhoods on 2D triangular grid.
        
        Each column is treated as a single node, and we find k-hop
        neighbors based on triangular connectivity.
        
        Returns: dict mapping column_id -> list of neighbor column_ids
        """
        if cached_neighborhoods is not None:
            return cached_neighborhoods
            
        # Build adjacency list from edge index (column-to-column connections)
        adj_list = {i: set() for i in range(num_columns)}
        
        # Extract horizontal connections between columns
        # Note: edge_index contains 3D edges, we need to extract 2D column connectivity
        for i in range(edge_index.size(1)):
            src, dst = edge_index[:, i].tolist()
            # Convert node indices to column indices
            # (assuming nodes are ordered as: col0_h0, col0_h1, ..., col1_h0, col1_h1, ...)
            src_col = src // num_height_levels
            dst_col = dst // num_height_levels
            
            # Only store column-to-column connections
            if src_col != dst_col and src_col < num_columns and dst_col < num_columns:
                adj_list[src_col].add(dst_col)
        
        # For each column, find k-hop neighboring columns
        column_neighborhoods = {}
        for col_id in range(num_columns):
            current_hop = {col_id}
            all_neighbor_cols = {col_id}
            
            for hop in range(self.max_hops):
                next_hop = set()
                for col in current_hop:
                    if col in adj_list:
                        next_hop.update(adj_list[col])
                current_hop = next_hop - all_neighbor_cols
                all_neighbor_cols.update(current_hop)
                
                if not current_hop:
                    break
            
            column_neighborhoods[col_id] = sorted(list(all_neighbor_cols))
        
        return column_neighborhoods
    
    def build_attention_mask(self, neighborhoods, B, num_columns, device):
        """
        Build efficient attention mask for vectorized computation.
        Returns padded neighbor indices and mask for 2D grid.
        """
        # Find maximum neighborhood size
        max_neighbors = max(len(neighbors) for neighbors in neighborhoods.values())
        
        # Build padded neighbor tensor
        padded_neighbors = torch.zeros(num_columns, max_neighbors, dtype=torch.long, device=device)
        neighbor_mask = torch.zeros(num_columns, max_neighbors, dtype=torch.bool, device=device)
        
        for col_idx in range(num_columns):
            neighbors = neighborhoods.get(col_idx, [col_idx])
            if not neighbors:
                neighbors = [col_idx]
                
            num_neighbors = len(neighbors)
            
            # Fill neighbor indices
            padded_neighbors[col_idx, :num_neighbors] = torch.tensor(neighbors, device=device)
            neighbor_mask[col_idx, :num_neighbors] = True
            
            # Pad remaining positions with self-reference (safe fallback)
            if num_neighbors < max_neighbors:
                padded_neighbors[col_idx, num_neighbors:] = col_idx
        
        # Expand for batch dimension
        padded_neighbors = padded_neighbors.unsqueeze(0).expand(B, -1, -1).contiguous()
        neighbor_mask = neighbor_mask.unsqueeze(0).expand(B, -1, -1).contiguous()
        
        return padded_neighbors, neighbor_mask, max_neighbors
    
    def forward(self, x, edge_index, num_columns, num_height_levels, shared_cache=None):
        """
        Forward pass with efficient vectorized attention on 2D grid.
        
        Args:
            x: [batch, num_columns, dim] node features (flattened vertical)
            edge_index: [2, num_edges] edge connectivity (3D edges)
            num_columns: number of columns in the grid
            num_height_levels: number of height levels (for converting 3D edges to 2D)
            shared_cache: shared caching for efficiency
        """
        B, N_columns, D = x.shape
        device = x.device
        
        # Use shared cache for neighborhoods (expensive computation)
        if shared_cache is not None:
            if shared_cache['neighborhoods'] is None:
                neighborhoods = self.get_2d_neighborhoods(edge_index, num_columns, num_height_levels)
                shared_cache['neighborhoods'] = neighborhoods
            
            if shared_cache['attention_mask'] is None:
                neighborhoods = shared_cache['neighborhoods']
                padded_neighbors, neighbor_mask, max_neighbors = self.build_attention_mask(
                    neighborhoods, B, num_columns, device
                )
                shared_cache['attention_mask'] = (padded_neighbors, neighbor_mask, max_neighbors)
            
            padded_neighbors, neighbor_mask, max_neighbors = shared_cache['attention_mask']
        else:
            raise ValueError("shared_cache must be provided for efficiency")
        
        # Project to Q, K, V
        q = self.to_q(x).view(B, N_columns, self.heads, self.dim_head)
        k = self.to_k(x).view(B, N_columns, self.heads, self.dim_head)
        v = self.to_v(x).view(B, N_columns, self.heads, self.dim_head)
        
        # Vectorized neighbor gathering
        neighbor_indices = padded_neighbors.unsqueeze(2).expand(-1, -1, self.heads, -1)
        batch_indices = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, N_columns, self.heads, max_neighbors)
        head_indices = torch.arange(self.heads, device=device).view(1, 1, self.heads, 1).expand(B, N_columns, self.heads, max_neighbors)
        
        # Gather neighbor K, V
        k_neighbors = k[batch_indices, neighbor_indices, head_indices]
        v_neighbors = v[batch_indices, neighbor_indices, head_indices]
        
        # Compute attention scores
        q_expanded = q.unsqueeze(3)  # [B, N_columns, heads, 1, dim_head]
        scores = torch.matmul(q_expanded, k_neighbors.transpose(-2, -1)) * self.scale
        scores = scores.squeeze(3)  # [B, N_columns, heads, max_neighbors]
        
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
        out = out.view(B, N_columns, -1)
        return self.to_out(out)


class GenCastStyleTransformerLayer(nn.Module):
    """
    GenCast-style transformer layer for 2D grid attention.
    Uses pre-normalization for better gradient flow.
    """
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout, max_hops):
        super().__init__()
        # Pre-normalization layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # GenCast-style attention
        self.attn = GenCastStyleAttention(dim, heads, dim_head, dropout, max_hops)
        
        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),  # Using GELU like in original transformers
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
    
    def forward(self, x, edge_index, num_columns, num_height_levels, shared_cache=None):
        # Pre-norm + attention + residual
        x = x + self.attn(self.norm1(x), edge_index, num_columns, num_height_levels, shared_cache)
        
        # Pre-norm + MLP + residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class GenCastStyleGraphTransformer3D(nn.Module):
    """
    GenCast-style Graph Transformer for 3D atmospheric data on ICON triangular grid.
    
    Key features:
    1. Flattens all height levels into node features (no explicit vertical dimension)
    2. Pure 2D attention on triangular grid with k-hop neighborhoods
    3. Similar to GenCast architecture but on ICON instead of icosahedral geometry
    4. Training pipeline compatibility
    
    This approach provides a direct comparison to the hybrid methods by treating
    atmospheric columns as single nodes with rich vertical feature encoding.
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
        
        # Calculate flattened input size
        # All height levels are concatenated into a single feature vector per column
        total_3d_features = channels_in_3d * num_height_levels
        total_input_features = total_3d_features + channels_in_2d
        
        # Input projection from flattened features to embedding dimension
        self.input_proj = nn.Linear(total_input_features, embed_dim)
        
        # Get actual number of columns
        triangle_indices = get_triangle_indices(triangle_id, division_factor, total_cols)
        actual_num_columns = len(triangle_indices)
        
        # 2D position embeddings (one per column)
        self.pos_embedding_2d = nn.Parameter(
            torch.randn(1, actual_num_columns, embed_dim) * 0.02
        )
        
        # GenCast-style transformer layers
        mlp_dim = int(embed_dim * mlp_ratio)
        self.layers = nn.ModuleList([
            GenCastStyleTransformerLayer(embed_dim, heads, dim_head, mlp_dim, dropout, max_hops)
            for _ in range(depth)
        ])
        
        # Output processing
        self.norm = nn.LayerNorm(embed_dim)
        
        # Output projection to flattened 3D output
        total_output_features = channels_out * num_height_levels
        self.output_proj = nn.Linear(embed_dim, total_output_features)
        
        # Initialize weights
        self._init_weights()
        
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
        B, N, L, C3d = x3d_norm.shape
        _, _, C2d = x2d_norm.shape
        
        # Get graph structure (cached) - we only need horizontal edges
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
        
        # Flatten vertical dimension into features (GenCast-style)
        x3d_flat = x3d_norm.view(B, N, L * C3d)  # [B, N, L*C3d]
        
        # Concatenate 3D and 2D features
        x_concat = torch.cat([x3d_flat, x2d_norm], dim=-1)  # [B, N, L*C3d + C2d]
        
        # Project to embedding space
        x = self.input_proj(x_concat)  # [B, N, embed_dim]
        
        # Add 2D position embeddings
        pos_emb_size = min(x.size(1), self.pos_embedding_2d.size(1))
        x[:, :pos_emb_size, :] += self.pos_embedding_2d[:, :pos_emb_size, :]
        
        # Process through GenCast-style transformer layers
        for layer in self.layers:
            x = layer(x, edge_index, num_columns, self.num_height_levels, self._shared_cache)
        
        # Final processing
        x = self.norm(x)
        x_out = self.output_proj(x)  # [B, N, L*channels_out]
        
        # Reshape back to 3D structure
        output = x_out.view(B, N, L, self.channels_out)  # [B, N, L, channels_out]
        
        return output




# Example usage and testing
if __name__ == "__main__":
    print("GenCast-Style Graph Transformer for ICON Triangular Grid")
    print("=" * 60)
    print("Key features:")
    print("1. Flattens all height levels into node features")
    print("2. Pure 2D attention on triangular grid")
    print("3. k-hop neighborhood attention")
    print("4. Similar to GenCast but on ICON geometry")
    print()
    
