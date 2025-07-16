import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.graph_3d import get_3d_graph
from utils.data_utils import get_triangle_indices


class VerticalAttention(nn.Module):
    """
    Vertical-only attention mechanism for full atmospheric column interaction.
    Each column processes independently with full height-level attention.
    """
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        
        assert inner_dim == dim, (
            f"Inner dimension (heads × dim_head = {heads} × {dim_head} = {inner_dim}) "
            f"must match embedding dimension ({dim})"
        )
        
        # Node feature projections
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
    
    def forward(self, x, num_columns, num_height_levels):
        """
        Apply vertical attention within each column independently.
        
        Args:
            x: [batch, num_nodes, dim] where num_nodes = num_columns * num_height_levels
            num_columns: number of atmospheric columns
            num_height_levels: number of height levels per column
        """
        B, N, D = x.shape
        
        # Reshape to separate columns and heights: [B, num_columns, num_height_levels, D]
        x_reshaped = x.view(B, num_columns, num_height_levels, D)
        
        # Project to Q, K, V
        q = self.to_q(x_reshaped).view(B, num_columns, num_height_levels, self.heads, self.dim_head)
        k = self.to_k(x_reshaped).view(B, num_columns, num_height_levels, self.heads, self.dim_head)
        v = self.to_v(x_reshaped).view(B, num_columns, num_height_levels, self.heads, self.dim_head)
        
        # Transpose for attention: [B, num_columns, heads, num_height_levels, dim_head]
        q = q.transpose(2, 3)
        k = k.transpose(2, 3)
        v = v.transpose(2, 3)
        
        # Compute attention scores within each column
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, num_columns, heads, H, H]
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # [B, num_columns, heads, H, dim_head]
        
        # Reshape back: [B, num_columns, num_height_levels, heads * dim_head]
        out = out.transpose(2, 3).contiguous().view(B, num_columns, num_height_levels, -1)
        
        # Project output and flatten back to original shape
        out = self.to_out(out)
        return out.view(B, N, D)


class HorizontalAttention(nn.Module):
    """
    Horizontal-only attention mechanism for same-height level interaction.
    Each height level processes independently with k-hop horizontal neighbors.
    """
    def __init__(self, dim, heads, dim_head, dropout, max_hops):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.max_hops = max_hops
        inner_dim = dim_head * heads
        
        assert inner_dim == dim, (
            f"Inner dimension (heads × dim_head = {heads} × {dim_head} = {inner_dim}) "
            f"must match embedding dimension ({dim})"
        )
        
        # Node feature projections
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
    
    def get_horizontal_neighborhoods(self, edge_index, num_columns, cached_neighborhoods=None):
        """Get k-hop column neighborhoods for horizontal attention."""
        if cached_neighborhoods is not None:
            return cached_neighborhoods
            
        # Build adjacency list from edge index
        adj_list = {i: set() for i in range(num_columns)}
        
        # Extract horizontal connections between columns (assuming edge_index represents column connectivity)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[:, i].tolist()
            # Convert node indices to column indices
            src_col = src // 70  # Assuming 70 height levels
            dst_col = dst // 70
            
            if 0 <= src_col < num_columns and 0 <= dst_col < num_columns and src_col != dst_col:
                adj_list[src_col].add(dst_col)
        
        # For each column, find k-hop neighboring columns
        column_neighborhoods = {}
        for col_id in range(num_columns):
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
        
        return column_neighborhoods
    
    def build_horizontal_attention_mask(self, column_neighborhoods, num_columns, device):
        """Build attention mask for horizontal neighbors."""
        max_neighbors = max(len(neighbors) for neighbors in column_neighborhoods.values())
        
        padded_neighbors = torch.zeros(num_columns, max_neighbors, dtype=torch.long, device=device)
        neighbor_mask = torch.zeros(num_columns, max_neighbors, dtype=torch.bool, device=device)
        
        for col_idx in range(num_columns):
            neighbors = column_neighborhoods.get(col_idx, [col_idx])
            if not neighbors:
                neighbors = [col_idx]
                
            num_neighbors = len(neighbors)
            padded_neighbors[col_idx, :num_neighbors] = torch.tensor(neighbors, device=device)
            neighbor_mask[col_idx, :num_neighbors] = True
            
            # Pad remaining positions with self-reference
            if num_neighbors < max_neighbors:
                padded_neighbors[col_idx, num_neighbors:] = col_idx
        
        return padded_neighbors, neighbor_mask, max_neighbors
    
    def forward(self, x, edge_index, num_columns, num_height_levels, shared_cache=None):
        """
        Apply horizontal attention at each height level independently.
        
        Args:
            x: [batch, num_nodes, dim]
            edge_index: [2, num_edges] 
            num_columns: number of atmospheric columns
            num_height_levels: number of height levels
            shared_cache: shared cache for neighborhoods
        """
        B, N, D = x.shape
        device = x.device
        
        # Get or compute horizontal neighborhoods
        cache_key = 'horizontal_neighborhoods'
        if shared_cache is not None:
            if shared_cache.get(cache_key) is None:
                # Only compute on single batch edge index
                single_batch_edges = edge_index[:, :edge_index.shape[1] // B] if B > 1 else edge_index
                column_neighborhoods = self.get_horizontal_neighborhoods(single_batch_edges, num_columns)
                shared_cache[cache_key] = column_neighborhoods
            
            if shared_cache.get('horizontal_mask') is None:
                column_neighborhoods = shared_cache[cache_key]
                padded_neighbors, neighbor_mask, max_neighbors = self.build_horizontal_attention_mask(
                    column_neighborhoods, num_columns, device
                )
                shared_cache['horizontal_mask'] = (padded_neighbors, neighbor_mask, max_neighbors)
            
            padded_neighbors, neighbor_mask, max_neighbors = shared_cache['horizontal_mask']
        else:
            raise ValueError("shared_cache must be provided for efficiency")
        
        # Reshape to separate columns and heights: [B, num_columns, num_height_levels, D]
        x_reshaped = x.view(B, num_columns, num_height_levels, D)
        
        # Project to Q, K, V
        q = self.to_q(x_reshaped).view(B, num_columns, num_height_levels, self.heads, self.dim_head)
        k = self.to_k(x_reshaped).view(B, num_columns, num_height_levels, self.heads, self.dim_head)
        v = self.to_v(x_reshaped).view(B, num_columns, num_height_levels, self.heads, self.dim_head)
        
        # Process each height level independently
        outputs = []
        for h in range(num_height_levels):
            # Get features for this height level: [B, num_columns, heads, dim_head]
            q_h = q[:, :, h, :, :]  
            k_h = k[:, :, h, :, :]
            v_h = v[:, :, h, :, :]
            
            # Gather neighbor features for this height level
            neighbor_indices = padded_neighbors.unsqueeze(0).unsqueeze(2).expand(B, -1, self.heads, -1)
            batch_indices = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, num_columns, self.heads, max_neighbors)
            head_indices = torch.arange(self.heads, device=device).view(1, 1, self.heads, 1).expand(B, num_columns, self.heads, max_neighbors)
            
            k_neighbors = k_h[batch_indices, neighbor_indices, head_indices]  # [B, num_columns, heads, max_neighbors, dim_head]
            v_neighbors = v_h[batch_indices, neighbor_indices, head_indices]
            
            # Compute attention scores
            q_expanded = q_h.unsqueeze(3)  # [B, num_columns, heads, 1, dim_head]
            scores = torch.matmul(q_expanded, k_neighbors.transpose(-2, -1)) * self.scale
            scores = scores.squeeze(3)  # [B, num_columns, heads, max_neighbors]
            
            # Apply mask
            mask_value = -1e9
            neighbor_mask_expanded = neighbor_mask.unsqueeze(0).unsqueeze(2)  # [1, num_columns, 1, max_neighbors]
            scores = scores.masked_fill(~neighbor_mask_expanded, mask_value)
            
            # Attention weights and output
            attn_weights = F.softmax(scores, dim=-1)  # [B, num_columns, heads, max_neighbors]
            
            # Apply attention to values
            attn_weights_expanded = attn_weights.unsqueeze(-1)  # [B, num_columns, heads, max_neighbors, 1]
            out_h = torch.sum(attn_weights_expanded * v_neighbors, dim=3)  # [B, num_columns, heads, dim_head]
            
            outputs.append(out_h)
        
        # Stack outputs for all height levels: [B, num_columns, num_height_levels, heads, dim_head]
        out = torch.stack(outputs, dim=2)
        
        # Reshape and project output
        out = out.view(B, num_columns, num_height_levels, -1)
        out = self.to_out(out)
        
        return out.view(B, N, D)


class AlternatingTransformerLayer(nn.Module):
    """
    Transformer layer that uses either vertical-only or horizontal-only attention.
    """
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout, max_hops, layer_type='vertical'):
        super().__init__()
        self.layer_type = layer_type
        
        # Pre-normalization layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Choose attention type
        if layer_type == 'vertical':
            self.attn = VerticalAttention(dim, heads, dim_head, dropout)
        elif layer_type == 'horizontal':
            self.attn = HorizontalAttention(dim, heads, dim_head, dropout, max_hops)
        else:
            raise ValueError(f"layer_type must be 'vertical' or 'horizontal', got {layer_type}")
        
        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.SiLU(), 
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim)
        )
    
    def forward(self, x, edge_index, num_columns, num_height_levels, shared_cache=None):
        # Pre-norm + attention + residual
        if self.layer_type == 'vertical':
            x = x + self.attn(self.norm1(x), num_columns, num_height_levels)
        else:  # horizontal
            x = x + self.attn(self.norm1(x), edge_index, num_columns, num_height_levels, shared_cache)
        
        # Pre-norm + MLP + residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class AlternatingGraphTransformer3D(nn.Module):
    """
    Graph Transformer that alternates between vertical and horizontal attention layers.
    Follows the ArchesWeather pattern of separating vertical and horizontal processing.
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
                 use_height_dependent_decoder=False,
                 start_with_vertical=True,  # Whether to start with vertical or horizontal layer
                 *args,
                 **kwargs):
        super().__init__()
        
        self.device = device
        self.channels_out = channels_out
        self.embed_dim = embed_dim
        self.num_height_levels = num_height_levels
        self.use_height_dependent_decoder = use_height_dependent_decoder
        
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
        
        # Positional embeddings
        self.pos_embedding_spatial = nn.Embedding(actual_num_columns, embed_dim)
        self.pos_embedding_height = nn.Embedding(num_height_levels, embed_dim)
        
        # Alternating transformer layers
        mlp_dim = int(embed_dim * mlp_ratio)
        self.layers = nn.ModuleList()
        
        for i in range(depth):
            if start_with_vertical:
                layer_type = 'vertical' if i % 2 == 0 else 'horizontal'
            else:
                layer_type = 'horizontal' if i % 2 == 0 else 'vertical'
            
            # Skip horizontal layers if disabled
            if layer_type == 'horizontal' and disable_horizontal:
                layer_type = 'vertical'
            
            self.layers.append(
                AlternatingTransformerLayer(embed_dim, heads, dim_head, mlp_dim, dropout, max_hops, layer_type)
            )
        
        # Output processing
        self.norm = nn.LayerNorm(embed_dim)
        
        # Choose decoder type
        if use_height_dependent_decoder:
            self.height_decoders = nn.ModuleList([
                nn.Linear(embed_dim, channels_out)
                for _ in range(num_height_levels)
            ])
        else:
            self.output_proj = nn.Linear(embed_dim, channels_out)
        
        # Shared cache for efficiency
        self._shared_cache = {
            'horizontal_neighborhoods': None,
            'horizontal_mask': None
        }
    
    def forward(self, x3d_norm, x2d_norm):
        """
        Forward pass with alternating vertical and horizontal attention layers.
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
        
        # Add positional embeddings
        spatial_indices = torch.arange(N, device=x.device).repeat_interleave(L)
        height_indices = torch.arange(L, device=x.device).repeat(N)
        
        pos_emb_spatial = self.pos_embedding_spatial(spatial_indices)
        pos_emb_height = self.pos_embedding_height(height_indices)
        
        x_flat = x_flat + pos_emb_spatial.unsqueeze(0) + pos_emb_height.unsqueeze(0)
        
        # Process through alternating transformer layers
        for layer in self.layers:
            x_flat = layer(x_flat, edge_index, num_columns, self.num_height_levels, self._shared_cache)
        
        # Reshape back to 3D structure
        x = x_flat.view(B, N, L, -1)
        
        # Final processing
        x = self.norm(x)
        
        # Apply decoder
        if self.use_height_dependent_decoder:
            outputs = []
            for level in range(self.num_height_levels):
                level_features = x[:, :, level, :]
                level_output = self.height_decoders[level](level_features)
                outputs.append(level_output)
            x = torch.stack(outputs, dim=2)
        else:
            x = self.output_proj(x)
        
        return x 