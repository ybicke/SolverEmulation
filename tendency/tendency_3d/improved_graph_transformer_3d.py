import torch
import torch.nn as nn
import torch.nn.functional as F
from graph_3d_full import get_3d_graph
from data_utils import get_triangle_indices
import math


class SpatialFeatureEncoder(nn.Module):
    """
    Encode spatial features similar to GraphCast's spatial feature encoding.
    Includes relative positions, distances, and directional features.
    """
    def __init__(self, dim, use_relative_positions=True, use_height_encoding=True):
        super().__init__()
        self.use_relative_positions = use_relative_positions
        self.use_height_encoding = use_height_encoding
        
        feature_dims = []
        if use_relative_positions:
            feature_dims.append(3)  # 3D relative positions (x, y, z or lat, lon, height)
        if use_height_encoding:
            feature_dims.append(2)  # Height level encoding (sin/cos)
            
        total_spatial_dim = sum(feature_dims) if feature_dims else 0
        
        if total_spatial_dim > 0:
            self.spatial_encoder = nn.Sequential(
                nn.Linear(total_spatial_dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim)
            )
        else:
            self.spatial_encoder = None
    
    def encode_height_features(self, height_levels, max_height):
        """Sinusoidal position encoding for height levels"""
        # Similar to GraphCast's positional encodings
        positions = height_levels.float() / max_height
        div_term = torch.exp(torch.arange(0, 2, 2).float() * 
                           -(math.log(10000.0) / 2))
        
        pe = torch.zeros(len(height_levels), 2)
        pe[:, 0] = torch.sin(positions.unsqueeze(1) * div_term).squeeze()
        pe[:, 1] = torch.cos(positions.unsqueeze(1) * div_term).squeeze()
        return pe
    
    def forward(self, node_features, spatial_coords=None, height_levels=None):
        if self.spatial_encoder is None:
            return node_features
            
        spatial_features = []
        
        if self.use_relative_positions and spatial_coords is not None:
            spatial_features.append(spatial_coords)
            
        if self.use_height_encoding and height_levels is not None:
            height_enc = self.encode_height_features(height_levels, height_levels.max())
            spatial_features.append(height_enc)
        
        if spatial_features:
            spatial_concat = torch.cat(spatial_features, dim=-1)
            spatial_encoded = self.spatial_encoder(spatial_concat)
            return node_features + spatial_encoded
        
        return node_features


class NormConditionedLinear(nn.Module):
    """
    Linear layer with normalization conditioning, similar to GraphCast's approach.
    This allows conditioning the model on global context (like noise levels in diffusion).
    """
    def __init__(self, in_features, out_features, conditioning_dim=None):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.layer_norm = nn.LayerNorm(out_features, elementwise_affine=False)
        
        if conditioning_dim is not None:
            # Learn scale and offset from conditioning
            self.conditioning_proj = nn.Linear(conditioning_dim, 2 * out_features)
        else:
            self.conditioning_proj = None
            
    def forward(self, x, conditioning=None):
        x = self.linear(x)
        x = self.layer_norm(x)
        
        if self.conditioning_proj is not None and conditioning is not None:
            # Apply learned scale and offset based on conditioning
            scale_offset = self.conditioning_proj(conditioning)
            scale, offset = scale_offset.chunk(2, dim=-1)
            if conditioning.dim() == 2:  # [batch, features]
                scale = scale.unsqueeze(1)  # [batch, 1, features]
                offset = offset.unsqueeze(1)
            x = x * (1 + scale) + offset
            
        return x


class ImprovedNeighborhoodAttention(nn.Module):
    """
    Enhanced neighborhood attention with features from GraphCast:
    1. Better edge feature integration
    2. Normalization conditioning
    3. Residual connections in attention
    4. Improved initialization
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, max_hops=2, 
                 conditioning_dim=None, use_edge_bias=True):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.max_hops = max_hops
        self.use_edge_bias = use_edge_bias
        inner_dim = dim_head * heads
        
        # Q, K, V projections with better initialization
        self.to_q = NormConditionedLinear(dim, inner_dim, conditioning_dim)
        self.to_k = NormConditionedLinear(dim, inner_dim, conditioning_dim)
        self.to_v = NormConditionedLinear(dim, inner_dim, conditioning_dim)
        
        # Edge bias for incorporating edge features into attention
        if use_edge_bias:
            self.edge_bias_proj = nn.Sequential(
                nn.Linear(dim, heads),
                nn.Tanh()  # Bounded bias values
            )
        
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Initialize weights similar to GraphCast
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with variance scaling similar to GraphCast"""
        for module in [self.to_q.linear, self.to_k.linear, self.to_v.linear]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        # Output projection initialized smaller to stabilize training
        nn.init.xavier_uniform_(self.to_out[0].weight, gain=0.1)
        if self.to_out[0].bias is not None:
            nn.init.zeros_(self.to_out[0].bias)
    
    def get_k_hop_neighbors(self, edge_index, num_nodes, k_hops, cached_neighborhoods=None):
        """Same as your original implementation"""
        if cached_neighborhoods is None:
            adj_list = {i: set() for i in range(num_nodes)}
            for i in range(edge_index.size(1)):
                src, dst = edge_index[:, i].tolist()
                if 0 <= src < num_nodes and 0 <= dst < num_nodes:
                    adj_list[src].add(dst)
            
            neighborhoods = {}
            for node in range(num_nodes):
                current_hop = {node}
                all_neighbors = {node}
                
                for hop in range(k_hops):
                    next_hop = set()
                    for n in current_hop:
                        next_hop.update(adj_list[n])
                    current_hop = next_hop - all_neighbors
                    all_neighbors.update(current_hop)
                    
                    if not current_hop:
                        break
                
                neighborhoods[node] = sorted(list(all_neighbors))
            
            cached_neighborhoods = neighborhoods
        
        return cached_neighborhoods
    
    def build_padded_neighborhoods(self, neighborhoods, B, N_per_batch, max_neighbors):
        """Same as your original implementation"""
        single_batch_neighbors = torch.zeros(N_per_batch, max_neighbors, dtype=torch.long)
        single_batch_mask = torch.zeros(N_per_batch, max_neighbors, dtype=torch.bool)
        
        for node_idx in range(N_per_batch):
            neighbors = neighborhoods.get(node_idx, [node_idx])
            if not neighbors:
                neighbors = [node_idx]
                
            num_neighbors = len(neighbors)
            single_batch_neighbors[node_idx, :num_neighbors] = torch.tensor(neighbors)
            single_batch_mask[node_idx, :num_neighbors] = True
            
            if num_neighbors < max_neighbors:
                single_batch_neighbors[node_idx, num_neighbors:] = node_idx
        
        padded_neighbors = single_batch_neighbors.unsqueeze(0).expand(B, -1, -1).clone()
        neighbor_mask = single_batch_mask.unsqueeze(0).expand(B, -1, -1).clone()
        
        return padded_neighbors, neighbor_mask
    
    def forward(self, x, edge_index, num_columns, shared_cache=None, 
                conditioning=None, edge_features=None):
        B, N_per_batch, D = x.shape
        device = x.device
        
        # Get neighborhoods (same caching logic as before)
        if shared_cache is not None:
            if shared_cache['neighborhoods'] is None:
                single_batch_edges = edge_index[:, :edge_index.shape[1] // B] if B > 1 else edge_index
                neighborhoods = self.get_k_hop_neighbors(
                    single_batch_edges, N_per_batch, self.max_hops, 
                    shared_cache['neighborhoods']
                )
                shared_cache['neighborhoods'] = neighborhoods
            
            if shared_cache['padded_neighbors'] is None:
                neighborhoods = shared_cache['neighborhoods']
                max_neighbors = max(len(neighbors) for neighbors in neighborhoods.values())
                
                padded_neighbors, neighbor_mask = self.build_padded_neighborhoods(
                    neighborhoods, B, N_per_batch, max_neighbors
                )
                shared_cache['padded_neighbors'] = padded_neighbors.to(device)
                shared_cache['neighbor_mask'] = neighbor_mask.to(device)
                shared_cache['max_neighbors'] = max_neighbors
            
            padded_neighbors = shared_cache['padded_neighbors']
            neighbor_mask = shared_cache['neighbor_mask']
            max_neighbors = shared_cache['max_neighbors']
        else:
            raise ValueError("shared_cache must be provided")
        
        # Project Q, K, V with conditioning
        q = self.to_q(x, conditioning)
        k = self.to_k(x, conditioning)
        v = self.to_v(x, conditioning)
        
        # Reshape for multi-head attention
        q = q.view(B, N_per_batch, self.heads, self.dim_head)
        k = k.view(B, N_per_batch, self.heads, self.dim_head)
        v = v.view(B, N_per_batch, self.heads, self.dim_head)
        
        # Gather neighbor features
        neighbor_indices = padded_neighbors.unsqueeze(2).expand(-1, -1, self.heads, -1)
        batch_indices = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, N_per_batch, self.heads, max_neighbors)
        head_indices = torch.arange(self.heads, device=device).view(1, 1, self.heads, 1).expand(B, N_per_batch, self.heads, max_neighbors)
        
        k_neighbors = k[batch_indices, neighbor_indices, head_indices]
        v_neighbors = v[batch_indices, neighbor_indices, head_indices]
        
        # Compute attention scores
        q_expanded = q.unsqueeze(3)
        scores = torch.matmul(q_expanded, k_neighbors.transpose(-2, -1)) * self.scale
        scores = scores.squeeze(3)
        
        # Add edge bias if available
        if self.use_edge_bias and edge_features is not None:
            # This would require edge features aligned with the attention pattern
            # For now, we'll skip this but it's a potential enhancement
            pass
        
        # Apply mask and softmax
        mask_value = -1e9
        neighbor_mask_expanded = neighbor_mask.unsqueeze(2)
        scores = scores.masked_fill(~neighbor_mask_expanded, mask_value)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_weights_expanded = attn_weights.unsqueeze(-1)
        out = torch.sum(attn_weights_expanded * v_neighbors, dim=3)
        
        # Reshape and project output
        out = out.view(B, N_per_batch, -1)
        return self.to_out(out)


class ImprovedGraphTransformerLayer(nn.Module):
    """
    Enhanced transformer layer with features from GraphCast:
    1. Pre-normalization instead of post-normalization
    2. Normalization conditioning
    3. Better residual connections
    4. Gated activation in MLP
    """
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0, max_hops=2, 
                 conditioning_dim=None, use_gated_mlp=True):
        super().__init__()
        
        # Pre-normalization (better for deep networks)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        
        # Attention with conditioning
        self.attn = ImprovedNeighborhoodAttention(
            dim, heads, dim_head, dropout, max_hops, conditioning_dim
        )
        
        # Enhanced MLP with gating (like SwiGLU)
        if use_gated_mlp:
            self.mlp = nn.Sequential(
                NormConditionedLinear(dim, mlp_dim * 2, conditioning_dim),
                nn.GELU(),  # Apply GELU to first half
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, dim),  # Gate and project
                nn.Dropout(dropout)
            )
        else:
            self.mlp = nn.Sequential(
                NormConditionedLinear(dim, mlp_dim, conditioning_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                NormConditionedLinear(mlp_dim, dim, conditioning_dim),
                nn.Dropout(dropout)
            )
        
        self.use_gated_mlp = use_gated_mlp
        
        # Conditioning projections for normalization
        if conditioning_dim is not None:
            self.norm1_conditioning = nn.Linear(conditioning_dim, 2 * dim)
            self.norm2_conditioning = nn.Linear(conditioning_dim, 2 * dim)
        else:
            self.norm1_conditioning = None
            self.norm2_conditioning = None
    
    def apply_conditioned_norm(self, x, norm_layer, conditioning_proj, conditioning):
        """Apply layer norm with conditioning"""
        x_norm = norm_layer(x)
        
        if conditioning_proj is not None and conditioning is not None:
            scale_offset = conditioning_proj(conditioning)
            scale, offset = scale_offset.chunk(2, dim=-1)
            if conditioning.dim() == 2:
                scale = scale.unsqueeze(1)
                offset = offset.unsqueeze(1)
            x_norm = x_norm * (1 + scale) + offset
        
        return x_norm
    
    def forward(self, x, edge_index, num_columns, shared_cache=None, conditioning=None):
        # Pre-norm + attention + residual
        x_norm1 = self.apply_conditioned_norm(x, self.norm1, self.norm1_conditioning, conditioning)
        attn_out = self.attn(x_norm1, edge_index, num_columns, shared_cache, conditioning)
        x = x + attn_out
        
        # Pre-norm + MLP + residual
        x_norm2 = self.apply_conditioned_norm(x, self.norm2, self.norm2_conditioning, conditioning)
        
        if self.use_gated_mlp:
            # Gated MLP forward pass
            mlp_out = self.mlp[0](x_norm2, conditioning)  # NormConditionedLinear
            gate, value = mlp_out.chunk(2, dim=-1)
            mlp_out = self.mlp[1](gate) * value  # GELU(gate) * value
            mlp_out = self.mlp[2](mlp_out)  # Dropout
            mlp_out = self.mlp[3](mlp_out)  # Final linear
            mlp_out = self.mlp[4](mlp_out)  # Final dropout
        else:
            mlp_out = self.mlp(x_norm2)
            
        x = x + mlp_out
        return x


class ImprovedGraphTransformer3D(nn.Module):
    """
    Enhanced Graph Transformer inspired by GraphCast architecture.
    
    Key improvements:
    1. Multi-stage processing with different objectives
    2. Better spatial feature encoding
    3. Normalization conditioning for global context
    4. Enhanced attention mechanisms
    5. Better initialization and regularization
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
                 use_spatial_encoding=True,
                 use_conditioning=True,
                 conditioning_dim=32,
                 use_gated_mlp=True,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.device = device
        self.channels_out = channels_out
        self.embed_dim = embed_dim
        self.num_height_levels = num_height_levels
        self.use_conditioning = use_conditioning
        
        # Store grid parameters
        self.grid_file_path = grid_file_path
        self.triangle_id = triangle_id
        self.division_factor = division_factor
        self.total_cols = total_cols
        self.fully_connected = fully_connected
        self.disable_horizontal = disable_horizontal
        
        # Input projection with better initialization
        total_channels = channels_in_3d + channels_in_2d
        self.input_proj = nn.Linear(total_channels, embed_dim)
        nn.init.xavier_uniform_(self.input_proj.weight)
        
        # Spatial feature encoding
        if use_spatial_encoding:
            self.spatial_encoder = SpatialFeatureEncoder(embed_dim)
        else:
            self.spatial_encoder = None
        
        # Global conditioning (for things like time embeddings, noise levels, etc.)
        if use_conditioning:
            self.conditioning_encoder = nn.Sequential(
                nn.Linear(conditioning_dim, conditioning_dim),
                nn.GELU(),
                nn.Linear(conditioning_dim, conditioning_dim)
            )
        else:
            conditioning_dim = None
        
        # Get actual number of columns
        triangle_indices = get_triangle_indices(triangle_id, division_factor, total_cols)
        actual_num_columns = len(triangle_indices)
        
        # Position embeddings with better initialization
        self.pos_embedding_3d = nn.Parameter(torch.randn(1, actual_num_columns * num_height_levels, embed_dim) * 0.02)
        
        # Enhanced transformer layers
        mlp_dim = int(embed_dim * mlp_ratio)
        self.graph_layers = nn.ModuleList([
            ImprovedGraphTransformerLayer(
                embed_dim, heads, dim_head, mlp_dim, dropout, max_hops, 
                conditioning_dim, use_gated_mlp
            )
            for _ in range(depth)
        ])
        
        # Output processing with conditioning
        if use_conditioning:
            self.output_norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
            self.output_norm_conditioning = nn.Linear(conditioning_dim, 2 * embed_dim)
        else:
            self.output_norm = nn.LayerNorm(embed_dim)
            self.output_norm_conditioning = None
            
        self.output_proj = nn.Linear(embed_dim, channels_out)
        
        # Initialize output projection smaller for stability
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.1)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)
        
        # Caching
        self._cached_neighborhoods = None
        self._cached_padded_neighbors = None
        self._cached_neighbor_mask = None
        self._max_neighbors = None
        
    def _get_shared_cache(self):
        return {
            'neighborhoods': self._cached_neighborhoods,
            'padded_neighbors': self._cached_padded_neighbors,
            'neighbor_mask': self._cached_neighbor_mask,
            'max_neighbors': self._max_neighbors
        }
    
    def forward(self, x3d_norm, x2d_norm, global_conditioning=None):
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
        x = self.input_proj(x_concat)
        
        # Add spatial encoding if enabled
        if self.spatial_encoder is not None:
            # Create height level features for spatial encoding
            height_levels = torch.arange(L, device=self.device).repeat(N)
            x_flat = x.view(B, N * L, -1)
            x_flat = self.spatial_encoder(x_flat, height_levels=height_levels)
            x = x_flat.view(B, N, L, -1)
        
        # Flatten for graph processing
        x_flat = x.view(B, N * L, -1)
        
        # Add position embeddings
        pos_emb_size = min(x_flat.size(1), self.pos_embedding_3d.size(1))
        x_flat[:, :pos_emb_size, :] = x_flat[:, :pos_emb_size, :] + self.pos_embedding_3d[:, :pos_emb_size, :]
        
        # Process global conditioning
        if self.use_conditioning and global_conditioning is not None:
            conditioning = self.conditioning_encoder(global_conditioning)
        else:
            conditioning = None
        
        # Graph transformer processing with conditioning
        shared_cache = self._get_shared_cache()
        for layer in self.graph_layers:
            x_flat = layer(x_flat, edge_index, num_columns, shared_cache, conditioning)
        
        # Reshape back to 3D
        x = x_flat.view(B, N, L, -1)
        
        # Final processing with conditioning
        if self.output_norm_conditioning is not None and conditioning is not None:
            x = self.output_norm(x)
            scale_offset = self.output_norm_conditioning(conditioning)
            scale, offset = scale_offset.chunk(2, dim=-1)
            scale = scale.view(B, 1, 1, -1)
            offset = offset.view(B, 1, 1, -1)
            x = x * (1 + scale) + offset
        else:
            x = self.output_norm(x)
        
        x = self.output_proj(x)
        
        return x 