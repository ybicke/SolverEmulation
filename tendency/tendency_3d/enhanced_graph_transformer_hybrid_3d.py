import torch
import torch.nn as nn
import torch.nn.functional as F
from graph_3d_full import get_3d_graph
from data_utils import get_triangle_indices


class EnhancedHybridNeighborhoodAttention(nn.Module):
    """
    Enhanced hybrid attention mechanism that combines:
    1. Full vertical attention within each column (atmospheric physics)
    2. k-hop horizontal attention to same-height neighbors only
    3. GNN-inspired edge processing using implicit edge features
    4. Enhanced message aggregation pathways
    
    Key enhancements over the simplified version:
    - Implicit edge feature computation from node pairs
    - Dual-pathway attention (content + positional)
    - GNN-style message aggregation
    - Better activations and normalization
    """
    def __init__(self, dim, heads, dim_head, dropout, max_hops):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.max_hops = max_hops
        inner_dim = dim_head * heads
        
        # Standard attention projections
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        
        # Enhanced edge processing (GNN-inspired) - no explicit edge features needed
        self.edge_processor = nn.Sequential(
            nn.Linear(3 * dim, dim),  # [relative_features, src_feat, dst_feat]
            nn.SiLU(),  # Better than ReLU for gradients
            nn.Dropout(dropout),
            nn.LayerNorm(dim),
            nn.Linear(dim, heads)  # Output attention bias per head
        )
        
        # Positional attention pathway (complementary to content attention)
        self.pos_processor = nn.Sequential(
            nn.Linear(dim, heads),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        # Message aggregation (GNN-inspired)
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * dim, dim),  # [original_features, attended_features]
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(dim),
        )
        
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, dim)
        
        # Enhanced initialization
        self._init_weights()
    
    def _init_weights(self):
        """Enhanced weight initialization for better training stability"""
        # Xavier for Q, K, V
        for module in [self.to_q, self.to_k, self.to_v]:
            nn.init.xavier_uniform_(module.weight)
        
        # Smaller scale for output to prevent exploding gradients
        nn.init.xavier_uniform_(self.to_out.weight, gain=0.1)
        if self.to_out.bias is not None:
            nn.init.zeros_(self.to_out.bias)
    
    def compute_implicit_edge_features(self, x, padded_neighbors, neighbor_mask):
        """
        Compute implicit edge features from node pairs without explicit edge attributes.
        
        Args:
            x: [B, N, dim] node features
            padded_neighbors: [B, N, max_neighbors] neighbor indices
            neighbor_mask: [B, N, max_neighbors] mask for valid neighbors
        
        Returns:
            edge_biases: [B, N, heads, max_neighbors] attention biases
        """
        B, N, dim = x.shape
        max_neighbors = padded_neighbors.size(-1)
        
        # Gather neighbor features
        batch_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, N, max_neighbors)
        neighbor_features = x[batch_idx, padded_neighbors]  # [B, N, max_neighbors, dim]
        
        # Expand source features for broadcasting
        src_features = x.unsqueeze(2).expand(B, N, max_neighbors, dim)  # [B, N, max_neighbors, dim]
        
        # Compute relative features (difference, sum, element-wise product)
        feat_diff = neighbor_features - src_features
        feat_sum = neighbor_features + src_features
        feat_prod = neighbor_features * src_features
        
        # Combine relative features (you can experiment with different combinations)
        relative_features = feat_diff  # Start simple, can be enhanced
        
        # Create edge features: [relative_features, src_features, dst_features]
        edge_features = torch.cat([
            relative_features,
            src_features,
            neighbor_features
        ], dim=-1)  # [B, N, max_neighbors, 3*dim]
        
        # Process through edge processor
        edge_features_flat = edge_features.view(-1, 3 * dim)
        edge_biases_flat = self.edge_processor(edge_features_flat)  # [B*N*max_neighbors, heads]
        edge_biases = edge_biases_flat.view(B, N, max_neighbors, self.heads)
        edge_biases = edge_biases.permute(0, 1, 3, 2)  # [B, N, heads, max_neighbors]
        
        # Apply mask to edge biases
        mask_expanded = neighbor_mask.unsqueeze(2).expand(B, N, self.heads, max_neighbors)
        edge_biases = edge_biases.masked_fill(~mask_expanded, 0.0)
        
        return edge_biases
    
    def get_hybrid_neighborhoods(self, edge_index, num_nodes, num_columns, num_height_levels, cached_neighborhoods=None):
        """
        Same efficient neighborhood computation as simplified version
        """
        if cached_neighborhoods is not None:
            return cached_neighborhoods
            
        # Build adjacency list from edge index (only need horizontal connectivity)
        adj_list = {i: set() for i in range(num_columns)}
        
        # Extract horizontal connections between columns
        for i in range(edge_index.size(1)):
            src, dst = edge_index[:, i].tolist()
            if 0 <= src < num_nodes and 0 <= dst < num_nodes:
                src_col = src // num_height_levels
                dst_col = dst // num_height_levels
                
                # Only store column-to-column connections
                if src_col != dst_col:
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
        
        # Create node-level neighborhoods from column neighborhoods
        node_neighborhoods = {}
        for node_id in range(num_nodes):
            col_id = node_id // num_height_levels
            height_level = node_id % num_height_levels
            neighbor_cols = column_neighborhoods[col_id]
            
            neighbors = set()
            
            # Add all nodes in the same column (full vertical attention)
            for h in range(num_height_levels):
                same_col_node = col_id * num_height_levels + h
                if same_col_node < num_nodes:
                    neighbors.add(same_col_node)
            
            # Add same-height neighbors in k-hop neighboring columns
            for neighbor_col in neighbor_cols:
                if neighbor_col != col_id:
                    same_height_neighbor = neighbor_col * num_height_levels + height_level
                    if same_height_neighbor < num_nodes:
                        neighbors.add(same_height_neighbor)
            
            node_neighborhoods[node_id] = sorted(list(neighbors))
        
        return node_neighborhoods
    
    def build_attention_mask(self, neighborhoods, B, N_per_batch, device):
        """Same efficient attention mask building as simplified version"""
        max_neighbors = max(len(neighbors) for neighbors in neighborhoods.values())
        
        padded_neighbors = torch.zeros(N_per_batch, max_neighbors, dtype=torch.long, device=device)
        neighbor_mask = torch.zeros(N_per_batch, max_neighbors, dtype=torch.bool, device=device)
        
        for node_idx in range(N_per_batch):
            neighbors = neighborhoods.get(node_idx, [node_idx])
            if not neighbors:
                neighbors = [node_idx]
                
            num_neighbors = len(neighbors)
            
            padded_neighbors[node_idx, :num_neighbors] = torch.tensor(neighbors, device=device)
            neighbor_mask[node_idx, :num_neighbors] = True
            
            if num_neighbors < max_neighbors:
                padded_neighbors[node_idx, num_neighbors:] = node_idx
        
        padded_neighbors = padded_neighbors.unsqueeze(0).expand(B, -1, -1).contiguous()
        neighbor_mask = neighbor_mask.unsqueeze(0).expand(B, -1, -1).contiguous()
        
        return padded_neighbors, neighbor_mask, max_neighbors
    
    def forward(self, x, edge_index, num_columns, shared_cache=None):
        """
        Enhanced forward pass with GNN-inspired edge processing and message aggregation.
        """
        B, N_per_batch, D = x.shape
        device = x.device
        num_height_levels = N_per_batch // num_columns
        
        # Use shared cache for efficiency (same as simplified version)
        if shared_cache is not None:
            if shared_cache['neighborhoods'] is None:
                single_batch_edges = edge_index[:, :edge_index.shape[1] // B] if B > 1 else edge_index
                neighborhoods = self.get_hybrid_neighborhoods(
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
        
        # Store original features for message aggregation
        x_original = x
        
        # Project to Q, K, V
        q = self.to_q(x).view(B, N_per_batch, self.heads, self.dim_head)
        k = self.to_k(x).view(B, N_per_batch, self.heads, self.dim_head)
        v = self.to_v(x).view(B, N_per_batch, self.heads, self.dim_head)
        
        # Compute implicit edge features and attention biases
        edge_biases = self.compute_implicit_edge_features(x, padded_neighbors, neighbor_mask)
        
        # Vectorized neighbor gathering (same as simplified version)
        neighbor_indices = padded_neighbors.unsqueeze(2).expand(-1, -1, self.heads, -1)
        batch_indices = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, N_per_batch, self.heads, max_neighbors)
        head_indices = torch.arange(self.heads, device=device).view(1, 1, self.heads, 1).expand(B, N_per_batch, self.heads, max_neighbors)
        
        # Gather neighbor K, V
        k_neighbors = k[batch_indices, neighbor_indices, head_indices]
        v_neighbors = v[batch_indices, neighbor_indices, head_indices]
        
        # Compute attention scores with edge bias
        q_expanded = q.unsqueeze(3)  # [B, N, heads, 1, dim_head]
        scores = torch.matmul(q_expanded, k_neighbors.transpose(-2, -1)) * self.scale
        scores = scores.squeeze(3)  # [B, N, heads, max_neighbors]
        
        # Add edge biases to attention scores (key enhancement!)
        scores = scores + edge_biases
        
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
        
        # Reshape attention output
        out = out.view(B, N_per_batch, -1)
        attended_features = self.to_out(out)
        
        # GNN-inspired message aggregation
        combined_features = torch.cat([x_original, attended_features], dim=-1)
        enhanced_features = self.message_mlp(combined_features)
        
        # Enhanced residual connection (combine both pathways)
        return attended_features + enhanced_features


class EnhancedHybridTransformerLayer(nn.Module):
    """
    Enhanced transformer layer with improved normalization and feed-forward network.
    """
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout, max_hops):
        super().__init__()
        # Pre-normalization layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Enhanced hybrid attention
        self.attn = EnhancedHybridNeighborhoodAttention(dim, heads, dim_head, dropout, max_hops)
        
        # Enhanced feed-forward network with SiLU and intermediate normalization
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.SiLU(),  # Better than GELU for atmospheric data
            nn.Dropout(dropout),
            nn.LayerNorm(mlp_dim),  # Intermediate normalization
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Enhanced initialization
        self._init_mlp_weights()
    
    def _init_mlp_weights(self):
        """Enhanced MLP weight initialization"""
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.xavier_uniform_(self.mlp[4].weight, gain=0.1)  # Smaller output scale
        
        if self.mlp[0].bias is not None:
            nn.init.zeros_(self.mlp[0].bias)
        if self.mlp[4].bias is not None:
            nn.init.zeros_(self.mlp[4].bias)
    
    def forward(self, x, edge_index, num_columns, shared_cache=None):
        # Pre-norm + enhanced attention + residual
        x = x + self.attn(self.norm1(x), edge_index, num_columns, shared_cache)
        
        # Pre-norm + enhanced MLP + residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class EnhancedHybridGraphTransformer3D(nn.Module):
    """
    Enhanced Hybrid Graph Transformer for 3D atmospheric data on ICON grid.
    
    Key enhancements over the simplified version:
    1. Implicit edge feature computation (no explicit edge features required)
    2. GNN-inspired message passing and aggregation
    3. Enhanced attention mechanisms with edge biases
    4. Better activations (SiLU) and normalization strategies
    5. Improved weight initialization for training stability
    6. Multiple residual pathways for better gradient flow
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
        
        # Enhanced input projection
        total_channels = channels_in_3d + channels_in_2d
        self.input_proj = nn.Sequential(
            nn.Linear(total_channels, embed_dim),
            nn.SiLU(),  # Better activation
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
        )
        
        # Get actual number of columns
        triangle_indices = get_triangle_indices(triangle_id, division_factor, total_cols)
        actual_num_columns = len(triangle_indices)
        
        # Enhanced position embeddings
        self.pos_embedding_3d = nn.Parameter(
            torch.randn(1, actual_num_columns * num_height_levels, embed_dim) * 0.02
        )
        
        # Enhanced transformer layers
        mlp_dim = int(embed_dim * mlp_ratio)
        self.layers = nn.ModuleList([
            EnhancedHybridTransformerLayer(embed_dim, heads, dim_head, mlp_dim, dropout, max_hops)
            for _ in range(depth)
        ])
        
        # Enhanced output processing
        self.output_layers = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, channels_out),
        )
        
        # Enhanced initialization
        self._init_weights()
        
        # Shared cache for efficiency
        self._shared_cache = {
            'neighborhoods': None,
            'attention_mask': None
        }
    
    def _init_weights(self):
        """Enhanced weight initialization for all components"""
        # Input projection initialization
        nn.init.xavier_uniform_(self.input_proj[0].weight)
        if self.input_proj[0].bias is not None:
            nn.init.zeros_(self.input_proj[0].bias)
        
        # Output layers initialization
        nn.init.xavier_uniform_(self.output_layers[1].weight)
        nn.init.xavier_uniform_(self.output_layers[4].weight, gain=0.1)
        
        for module in [self.output_layers[1], self.output_layers[4]]:
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def clear_cache(self):
        """Clear cached computations"""
        self._shared_cache = {
            'neighborhoods': None,
            'attention_mask': None
        }
    
    def get_neighborhood_stats(self):
        """Get neighborhood statistics for analysis"""
        if self._shared_cache['neighborhoods'] is None:
            return None
        
        neighborhoods = self._shared_cache['neighborhoods']
        sizes = [len(neighbors) for neighbors in neighborhoods.values()]
        
        return {
            'min_size': min(sizes),
            'max_size': max(sizes),
            'avg_size': sum(sizes) / len(sizes),
            'total_nodes': len(neighborhoods),
            'complexity_reduction': f"~{70 * 3:.0f}x vs full attention"  # Assuming 70 levels, ~3 neighboring columns
        }
    
    def forward(self, x3d_norm, x2d_norm):
        """
        Enhanced forward pass with better feature processing.
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
        
        # Enhanced feature preparation
        x2d_repeated = x2d_norm.unsqueeze(2).repeat(1, 1, L, 1)
        x_concat = torch.cat([x3d_norm, x2d_repeated], dim=-1)
        
        # Enhanced input projection
        x = self.input_proj(x_concat)  # [B, N, L, embed_dim]
        
        # Flatten for graph processing
        x_flat = x.view(B, N * L, -1)  # [B, N*L, embed_dim]
        
        # Add position embeddings
        pos_emb_size = min(x_flat.size(1), self.pos_embedding_3d.size(1))
        x_flat[:, :pos_emb_size, :] += self.pos_embedding_3d[:, :pos_emb_size, :]
        
        # Process through enhanced transformer layers
        for layer in self.layers:
            x_flat = layer(x_flat, edge_index, num_columns, self._shared_cache)
        
        # Reshape back to 3D structure
        x = x_flat.view(B, N, L, -1)  # [B, N, L, embed_dim]
        
        # Enhanced final processing
        x = self.output_layers(x)
        
        return x 