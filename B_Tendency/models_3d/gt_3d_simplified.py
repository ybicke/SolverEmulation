import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.graph_3d import get_3d_graph
from utils.data_utils import get_triangle_indices


class SimplifiedNeighborhoodAttention(nn.Module):
    """
    Hybrid attention mechanism that combines:
    1. Full vertical attention within each column (atmospheric physics)
    2. k-hop horizontal attention to same-height neighbors only
    
    This approach is physically motivated and computationally efficient:
    - Vertical: Full atmospheric column interaction (convection, radiation)
    - Horizontal: Same-level transport (pressure systems, advection)
    
    Neighborhood size: num_height_levels + k_hop_horizontal_neighbors
    vs original: num_height_levels × (1 + k_hop_neighboring_columns)
    
    This represents a ~4-6x reduction in attention complexity while maintaining
    the essential atmospheric physics interactions.
    """
    def __init__(self, dim, heads, dim_head, dropout, max_hops):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.max_hops = max_hops
        inner_dim = dim_head * heads
        
        # Ensure inner dimension matches embedding dimension
        assert inner_dim == dim, (
            f"Inner dimension (heads × dim_head = {heads} × {dim_head} = {inner_dim}) "
            f"must match embedding dimension ({dim})"
        )
        
        # Node feature projections
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        
        # Output projection only - no attention dropout for better performance
        self.to_out = nn.Linear(inner_dim, dim)
    
    def get_simplified_neighborhoods(self, edge_index, num_nodes, num_columns, num_height_levels, cached_neighborhoods=None):
        """
        Compute hybrid neighborhoods efficiently.
        
        For each node, the neighborhood includes:
        1. All nodes in the same column (full vertical attention)
        2. Same-height nodes in k-hop neighboring columns (horizontal neighbors at same level only)
        
        This is more efficient and physically motivated than including full vertical
        extent of neighboring columns.
        
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
                src_col = src // num_height_levels
                dst_col = dst // num_height_levels
                
                # Only store column-to-column connections that exist in the edge index 
                if src_col != dst_col:
                    adj_list[src_col].add(dst_col)
        
        # For each column, find k-hop neighboring columns
        column_neighborhoods = {}
        for col_id in range(num_columns):
            current_hop = {col_id}
            all_neighbor_cols = {col_id}
            
            for hop in range(self.max_hops):
                next_hop = set()  # Find direct neighbors of column in current_hop
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
                if neighbor_col != col_id:  # Skip self-column (already added above)
                    same_height_neighbor = neighbor_col * num_height_levels + height_level
                    if same_height_neighbor < num_nodes:
                        neighbors.add(same_height_neighbor)
            
            node_neighborhoods[node_id] = sorted(list(neighbors))
        
        return node_neighborhoods
    
    def build_attention_mask(self, neighborhoods, B, N_per_batch, device):
        """
        Build efficient attention mask for vectorized computation. Different nodes have different number of neighbors.
        Interior nodes for k=1, interior (70 vertical + 4 horizontal), edge (70 vertical + 2 horizontal), corner (70 vertical + 1 horizontal)
        Pytorch needs uniform tensor shapes for batch processing. Returns padded neighbor indices and mask.
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
                neighborhoods = self.get_simplified_neighborhoods(
                    single_batch_edges, N_per_batch, num_columns, num_height_levels
                ) # each node recieved it's neighborhood list
                shared_cache['neighborhoods'] = neighborhoods
            
            if shared_cache['attention_mask'] is None: # crate attention mask based on neighbourhood list
                neighborhoods = shared_cache['neighborhoods']
                padded_neighbors, neighbor_mask, max_neighbors = self.build_attention_mask(
                    neighborhoods, B, N_per_batch, device
                )
                shared_cache['attention_mask'] = (padded_neighbors, neighbor_mask, max_neighbors)
            # unified attention mask structure that contains the neighbor information for all node
            padded_neighbors, neighbor_mask, max_neighbors = shared_cache['attention_mask']
        else:
            raise ValueError("shared_cache must be provided for efficiency")
        
        # Project to Q, K, V
        q = self.to_q(x).view(B, N_per_batch, self.heads, self.dim_head)
        k = self.to_k(x).view(B, N_per_batch, self.heads, self.dim_head)
        v = self.to_v(x).view(B, N_per_batch, self.heads, self.dim_head)
        
        # Vectorized neighbor gathering for all nodes and all heads in a single vectorized operation
        neighbor_indices = padded_neighbors.unsqueeze(2).expand(-1, -1, self.heads, -1) # which neighbor to attend to, same for all heads
        batch_indices = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, N_per_batch, self.heads, max_neighbors) # which batch each lookup
        head_indices = torch.arange(self.heads, device=device).view(1, 1, self.heads, 1).expand(B, N_per_batch, self.heads, max_neighbors) # which head each lookup
        
        # Gather neighbor K, V
        k_neighbors = k[batch_indices, neighbor_indices, head_indices]
        v_neighbors = v[batch_indices, neighbor_indices, head_indices]
        
        # Compute attention scores
        q_expanded = q.unsqueeze(3)  # [B, N, heads, 1, dim_head]
        scores = torch.matmul(q_expanded, k_neighbors.transpose(-2, -1)) * self.scale
        scores = scores.squeeze(3)  # [B, N, heads, max_neighbors]
        
        # Apply mask
        mask_value = -1e9
        neighbor_mask_expanded = neighbor_mask.unsqueeze(2)
        scores = scores.masked_fill(~neighbor_mask_expanded, mask_value)
        
        # Attention weights and output - no dropout for better performance
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attn_weights_expanded = attn_weights.unsqueeze(-1)
        out = torch.sum(attn_weights_expanded * v_neighbors, dim=3)
        
        # Reshape and project output
        out = out.view(B, N_per_batch, -1)
        return self.to_out(out)


class SimplifiedTransformerLayer(nn.Module):
    """
    Transformer layer with hybrid attention and feed-forward network.
    Uses pre-normalization for better gradient flow.
    """
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout, max_hops):
        super().__init__()
        # Pre-normalization layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Hybrid attention without dropout
        self.attn = SimplifiedNeighborhoodAttention(dim, heads, dim_head, dropout, max_hops)
        
        # Feed-forward network with single dropout after activation
        # Dropout only after activation for minimal regularization
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.SiLU(), 
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim)
        )
    
    def forward(self, x, edge_index, num_columns, shared_cache=None):
        # Pre-norm + attention + residual
        x = x + self.attn(self.norm1(x), edge_index, num_columns, shared_cache)
        
        # Pre-norm + MLP + residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class SimplifiedGraphTransformer3D(nn.Module):
    """
    Hybrid Graph Transformer for 3D atmospheric data on ICON grid.
    
    Key features:
    1. Physically-motivated hybrid attention (full vertical + k-hop horizontal)
    2. Efficient vectorized implementation
    3. Proper caching for repeated computations
    4. Training pipeline compatibility
    5. Optional height-dependent decoder
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
                 use_height_dependent_decoder=False,  # NEW: For height-dependent decoder
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
        
        # Simplified positional embeddings - separate for spatial and height
        self.pos_embedding_spatial = nn.Embedding(actual_num_columns, embed_dim)  # Column positions
        self.pos_embedding_height = nn.Embedding(num_height_levels, embed_dim)   # Height positions
        
        # Hybrid transformer layers
        mlp_dim = int(embed_dim * mlp_ratio)
        self.layers = nn.ModuleList([
            SimplifiedTransformerLayer(embed_dim, heads, dim_head, mlp_dim, dropout, max_hops)
            for _ in range(depth)
        ])
        
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
        
       
        # Create position indices for embeddings
        # Spatial positions: column indices repeated for each height level. [0,0,...,0, 1,1,...,1, ..., N-1,N-1,...,N-1]
        spatial_indices = torch.arange(N, device=x.device).repeat_interleave(L)  
        # Height positions: height indices repeated for each column. [0,1,2,...,L-1, 0,1,2,...,L-1, ...]
         
        height_indices = torch.arange(L, device=x.device).repeat(N)  
        # Get positional embeddings
        pos_emb_spatial = self.pos_embedding_spatial(spatial_indices)  # [N*L, embed_dim]
        pos_emb_height = self.pos_embedding_height(height_indices)      # [N*L, embed_dim]
        
        # Add positional embeddings: token + pos_emb_spatial + pos_emb_height
        x_flat = x_flat + pos_emb_spatial.unsqueeze(0) + pos_emb_height.unsqueeze(0)  # [B, N*L, embed_dim]
        
        
        # Process through hybrid transformer layers
        for layer in self.layers:
            x_flat = layer(x_flat, edge_index, num_columns, self._shared_cache)
        
        # Reshape back to 3D structure
        x = x_flat.view(B, N, L, -1)  # [B, N, L, embed_dim]
        
        # Final processing
        x = self.norm(x)
        
        # Apply decoder (height-dependent or shared)
        if self.use_height_dependent_decoder:
            # Apply height-specific decoders
            outputs = []
            for level in range(self.num_height_levels):
                level_features = x[:, :, level, :]  # [B, N, embed_dim]
                level_output = self.height_decoders[level](level_features)  # [B, N, channels_out]
                outputs.append(level_output)
            
            # Stack to final output shape: [B, N, L, channels_out]
            x = torch.stack(outputs, dim=2)
        else:
            # Use shared decoder
            x = self.output_proj(x)
        
        return x 