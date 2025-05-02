import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import torch.utils.checkpoint

from .base_methods import BaseRadiationModel



class AtmosphericColumnGNN(BaseRadiationModel):
    """
    Graph Neural Network for atmospheric column radiative transfer modeling.
    Memory-optimized version with edge index caching and efficient processing.
    """
    def __init__(self,
                 embed_dim,
                 depth,
                 dropout,
                 max_skip,
                 emb_dropout,
                 channel_3d,
                 channel_2d,
                 channels_out,
                 edge_channels_in,
                 fully_connected,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.channels_out = channels_out
        self.max_skip = max_skip
        self.embed_dim = embed_dim
        self.fully_connected = fully_connected
        self.edge_channels_in = edge_channels_in
        
        # Cache for edge indices to avoid redundant computation
        self.edge_index_cache = {}
        
        channels_in = channel_3d + channel_2d

        self.encoder = Encoder(channels_in, edge_channels_in, embed_dim, emb_dropout)
        self.processor = Processor(embed_dim, depth=depth, dropout=dropout)
        self.decoder = Decoder(embed_dim, channels_out, dropout=dropout)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x3d_norm, x2d_norm, x2d_orig):
        B, L, _ = x3d_norm.shape

        # Append surface features to each atmospheric level
        surface_features = x2d_norm.unsqueeze(1)
        repeat_surface_at_all_levels = surface_features.repeat(1, L, 1)
        augmented_atmospheric_column = torch.cat([x3d_norm, repeat_surface_at_all_levels], dim=-1)
        
        # Create extra surface nodes with zeros and match the batch dimension
        one_surface_tensor = torch.ones(B, 1, x2d_norm.shape[1], device=x3d_norm.device)
        append_one_surface_tensor = torch.cat([one_surface_tensor, surface_features], dim=-1)
        
        # Combine extra zero surface nodes with atmospheric columns
        x = torch.cat([augmented_atmospheric_column, append_one_surface_tensor], dim=1)
        
        # MEMORY OPTIMIZATION 1: Cache edge indices for common configurations
        cache_key = (L+1, B, self.fully_connected)
        if cache_key in self.edge_index_cache and self.edge_index_cache[cache_key].device == x.device:
            edge_index = self.edge_index_cache[cache_key]
        else:
            if self.fully_connected:
                edge_index = create_edge_index_fully_connected_fast(L+1, B, x.device)
            else:
                edge_index = create_edge_index_multiMesh(L+1, B, x.device, self.max_skip)
            
            # Only cache common configurations to prevent memory leaks
            if B in [32, 64, 128, 256, 512]:
                self.edge_index_cache[cache_key] = edge_index
        
        # Encode node features first
        x = self.encoder.node_mlp(x)
        B, N, E = x.shape
        x = x.view(B*N, E)
        
        # Save memory without embedding edge features with mlp's
        num_edges = edge_index.size(1)
        edge_attr = torch.zeros(num_edges, self.edge_channels_in, device=x.device)
        
        # Pass directly to processor, skipping edge encoding - now with fully_connected flag
        x = self.processor(x, edge_index, edge_attr, fully_connected=self.fully_connected)

        # reshape the output to the original shape and decode the output variables
        x = x.view(B, L+1, -1)
        x = self.decoder(x)
        
        # scale the output variables to the original range
        x = self.sigmoid(x)
        x = self._scale_output(x, x2d_orig)

        return x.squeeze()
    
    
class Encoder(nn.Module):
    """
    Input shape: [B, L+1, channels_in].
    Output shape: [B*(L+1), embed_dim].
    """
    def __init__(self, channels_in, edge_channels_in, embed_dim, emb_dropout):
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(channels_in, embed_dim),
            nn.SiLU(),
            nn.Dropout(emb_dropout),
            nn.LayerNorm(embed_dim),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_channels_in, embed_dim),
            nn.SiLU(), 
            nn.Dropout(emb_dropout),
            nn.LayerNorm(embed_dim),
        ) if edge_channels_in > 0 else None

    def forward(self, x, edge_attr):
        # Embed node features
        x = self.node_mlp(x)
        
        # Reshape node embeddings for GNN processing
        B, N, E = x.shape
        x = x.view(B*N, E)
        
        # Embed edge features if edge_mlp exists
        if self.edge_mlp is not None and edge_attr is not None:
            edge_attr = self.edge_mlp(edge_attr)

        return x, edge_attr
    
    


class GNNLayer(MessagePassing):
    def __init__(self, embed_dim, dropout):
        super().__init__(aggr='add')
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * embed_dim, embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
        )
        
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        # First, update all edges using equation 11
        # Extract source and target node features for each edge
        src, dst = edge_index
        x_src = x[src]
        x_dst = x[dst]
        
        # Compute edge updates (equation 11)
        edge_update = self.edge_mlp(torch.cat([edge_attr, x_src, x_dst], dim=-1))
        
        # Then, update all nodes by aggregating messages from incoming edges (equation 12)
        # propagate() handles the message passing and aggregation
        aggregated_edge_updates = self.propagate(edge_index, x=x, edge_attr=edge_update)
        
        # Concatenate node features with aggregated edge updates and apply MLP
        node_update = self.node_mlp(torch.cat([x, aggregated_edge_updates], dim=1))
        
        return node_update, edge_update
    
    def message(self, edge_attr):
        # Simply pass the updated edge features as messages
        return edge_attr
    
    def update(self, aggr_out):
        return aggr_out
    
    

    
class Processor(nn.Module):
    def __init__(self, embed_dim, depth, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            HybridGNNLayer(embed_dim=embed_dim, dropout=dropout)
            for _ in range(depth)
        ])
        self.use_dense_ops = True  # Flag to toggle dense optimization for fully connected graphs
        
    def forward(self, x, edge_index, edge_attr, fully_connected=None):
        """
        Forward pass that automatically selects the most efficient algorithm.
        
        Args:
            x: Node features [B*N, E] or [B, N, E]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, E]
            fully_connected: Whether the graph is fully connected (optional)
        """
        # Determine if we should use dense operations
        use_dense = self.use_dense_ops
        if fully_connected is not None:
            use_dense = fully_connected and self.use_dense_ops
        
        if use_dense:
            # For fully connected graph, use dense operations
            # Reshape x to [B, N, E] if it's not already
            if x.dim() == 2:
                # Estimate B and N from edge_index
                N = int((edge_index.max().item() + 1) ** 0.5)
                B = (edge_index.max().item() + 1) // N
                x_reshaped = x.view(B, N, -1)
            else:
                x_reshaped = x
            
            # Process with dense operations
            for layer in self.layers:
                node_update, _ = layer.forward_dense(x_reshaped)
                x_reshaped = x_reshaped + node_update
            
            # Reshape back to match input format
            if x.dim() == 2:
                return x_reshaped.view(B*N, -1)
            else:
                return x_reshaped
        else:
            # For sparse graph, use standard message passing
            for layer in self.layers:
                node_update, edge_update = layer.forward_sparse(x, edge_index, edge_attr)
                x = x + node_update
                edge_attr = edge_attr + edge_update
            
            return x
    
    

class Decoder(nn.Module):
    """
    Input shape (reshaped): [B, L+1, embed_dim].
    Output shape: [B, L+1, channels_out], 
    """
    def __init__(self, embed_dim, channels_out, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, channels_out),
            # TODO: could test with a layer norm here. Different from GraphCast.
        )

    def forward(self, x):
            B, N, E = x.shape
            x = x.view(B*N, E)
            x = self.mlp(x)
            x = x.view(B, N, self.mlp[-1].out_features)
            return x



def create_edge_index_multiMesh(num_nodes, batch_size, device, max_skip):
    """
    Creates a bidirectional edge index for a 1D chain with a hierarchical skip distance.
    
    Args:
        num_nodes (int): Number of nodes in a single chain.
        batch_size (int): Number of chains (batches).
        device (torch.device): The device to push edge_index onto.
        max_skip (int): Maximum skip distance (inclusive).
        
    Returns:
        edge_index (torch.LongTensor of shape [2, E]):
            Bidirectional edge indices for the 1D chain.
            
    Skip = 1:
    Forward: [0→1, 1→2, 2→3, 3→4, 4→5]
    Reverse: [1→0, 2→1, 3→2, 4→3, 5→4]

    Skip = 2:
    Forward: [0→2, 1→3, 2→4, 3→5]
    Reverse: [2→0, 3→1, 4→2, 5→3]

    Skip = 3:
    Forward: [0→3, 1→4, 2→5]
    Reverse: [3→0, 4→1, 5→2]
    
    edge_index = [
    [0,1,2,3,4, 1,2,3,4,5, 0,1,2,3, 2,3,4,5, 0,1,2, 3,4,5],  # source nodes
    [1,2,3,4,5, 0,1,2,3,4, 2,3,4,5, 0,1,2,3, 3,4,5, 0,1,2]   # target nodes
    ]
    """
    all_edges = []
    
    for skip in range(1, max_skip + 1):
        if skip >= num_nodes:
            break  # Skip distance is too large
            
        # Forward edges (i -> i + skip), first row contains the indices of the source nodes, 
        # second row contains the indices of the target nodes
        forward_edges = torch.stack([
            torch.arange(num_nodes - skip, device=device),
            torch.arange(skip, num_nodes, device=device)
        ], dim=0)
        
        # Reverse edges (i + skip -> i)
        reverse_edges = torch.stack([
            torch.arange(skip, num_nodes, device=device),
            torch.arange(num_nodes - skip, device=device)
        ], dim=0)
        
        # Combine forward and reverse
        base_edges = torch.cat([forward_edges, reverse_edges], dim=1)
        
        # Repeat for batch_size
        base_edges = base_edges.repeat(1, batch_size)
        
        # Offset node indices for each chain in the batch
        offset = torch.arange(batch_size, device=device) * num_nodes
        offset = offset.repeat_interleave(base_edges.shape[1] // batch_size)
        base_edges += offset.view(1, -1)
        
        all_edges.append(base_edges)
    
    # Concatenate edges from all skip distances
    edge_index = torch.cat(all_edges, dim=1)
    return edge_index


def create_edge_index_fully_connected(num_nodes, batch_size, device):
    """
    Creates a fully connected edge index where each node is connected to every other node,
    using highly vectorized tensor operations.
    """
    # For a single graph, generate all source nodes
    sources = torch.arange(num_nodes, device=device).repeat_interleave(num_nodes-1)
    
    # For each source, generate all targets (excluding self)
    target_ranges = []
    for i in range(num_nodes):
        # All nodes except self
        targets = torch.cat([
            torch.arange(0, i, device=device),
            torch.arange(i+1, num_nodes, device=device)
        ])
        target_ranges.append(targets)
    
    # Combine all targets
    targets = torch.cat(target_ranges)
    
    # Stack into a 2 x E tensor for a single batch
    base_edges = torch.stack([sources, targets])
    
    # Repeat for batch_size (just like in your hierarchical function)
    base_edges = base_edges.repeat(1, batch_size)
    
    # Offset node indices for each batch
    offset = torch.arange(batch_size, device=device) * num_nodes
    offset = offset.repeat_interleave(base_edges.shape[1] // batch_size)
    edge_index = base_edges + offset.view(1, -1)
    
    return edge_index




def create_edge_index_fully_connected_fast(num_nodes, batch_size, device):
    # Create edges for a single graph
    row = torch.arange(num_nodes, device=device)
    col = torch.arange(num_nodes, device=device)
    
    # Create a mesh grid of all possible connections
    row = row.repeat_interleave(num_nodes)
    col = col.repeat(num_nodes)
    
    # Remove self-loops
    mask = row != col
    row, col = row[mask], col[mask]
    
    # Stack into a 2 x E tensor for a single graph
    edge_index_single = torch.stack([row, col], dim=0)
    
    # Fast batch offsets
    edge_count = edge_index_single.size(1)
    batch_edge_index = edge_index_single.repeat(1, batch_size)
    
    # Create offsets tensor once
    offsets = torch.arange(0, batch_size, device=device) * num_nodes
    
    # Apply offsets to all batches at once (vectorized)
    batch_indices = torch.arange(batch_size, device=device).repeat_interleave(edge_count)
    batch_offsets = offsets[batch_indices]
    
    # Add offsets to both source and target nodes
    batch_edge_index[0] += batch_offsets
    batch_edge_index[1] += batch_offsets
    
    return batch_edge_index


class HybridGNNLayer(nn.Module):
    """
    A hybrid GNN layer that preserves the GraphCast architecture but 
    optimizes performance by using dense operations for fully connected graphs.
    """
    def __init__(self, embed_dim, dropout):
        super().__init__()
        
        # Use the same MLPs as the original GraphCast-inspired implementation
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * embed_dim, embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
        )
        
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
        )
        
    def forward_dense(self, x):
        """Dense implementation for fully connected graphs"""
        B, N, E = x.shape
        
        # Create zero edge features with the same dimensionality as node features
        edge_attr = torch.zeros(B, N, N, E, device=x.device)
        
        # Create pairs of source and target node features
        x_src = x.unsqueeze(2).expand(B, N, N, E)  # [B, N, N, E]
        x_dst = x.unsqueeze(1).expand(B, N, N, E)  # [B, N, N, E]
        
        # For smaller graphs, process all at once
        if B * N * N <= 1024 * 1024:  # Threshold for when to use batched processing
            # Concatenate edge features with node features
            combined = torch.cat([edge_attr, x_src, x_dst], dim=-1)
            
            # Apply edge MLP
            edge_updates = self.edge_mlp(combined.view(B*N*N, -1)).view(B, N, N, E)
            
            # Remove self-loops
            mask = ~torch.eye(N, dtype=torch.bool, device=x.device).view(1, N, N, 1)
            edge_updates = edge_updates * mask
            
            # Aggregate messages by summing across neighbor dimension
            aggregated = edge_updates.sum(dim=2)  # [B, N, E]
            
            # Compute node updates
            node_inputs = torch.cat([x, aggregated], dim=-1)  # [B, N, 2E]
            node_updates = self.node_mlp(node_inputs)  # [B, N, E]
            
            return node_updates, edge_updates
        
        # For larger graphs, use batched processing
        else:
            # Process in chunks to avoid excessive memory usage
            batch_size = 1024 * 32  # Adjust based on your GPU memory
            edge_updates = torch.zeros(B, N, N, E, device=x.device)
            
            # Flatten tensors for batch processing
            edge_attr_flat = edge_attr.reshape(B*N*N, E)
            x_src_flat = x_src.reshape(B*N*N, E)
            x_dst_flat = x_dst.reshape(B*N*N, E)
            
            # Process in batches
            for i in range(0, B*N*N, batch_size):
                end_idx = min(i + batch_size, B*N*N)
                
                # Extract batch data
                batch_edge_attr = edge_attr_flat[i:end_idx]
                batch_x_src = x_src_flat[i:end_idx]
                batch_x_dst = x_dst_flat[i:end_idx]
                
                # Process batch
                batch_combined = torch.cat([batch_edge_attr, batch_x_src, batch_x_dst], dim=-1)
                batch_result = self.edge_mlp(batch_combined)
                
                # Store results
                edge_updates.reshape(B*N*N, E)[i:end_idx] = batch_result
            
            # Remove self-loops
            mask = ~torch.eye(N, dtype=torch.bool, device=x.device).view(1, N, N, 1)
            edge_updates = edge_updates * mask
            
            # Aggregate messages
            aggregated = edge_updates.sum(dim=2)  # [B, N, E]
            
            # Compute node updates
            node_inputs = torch.cat([x, aggregated], dim=-1)  # [B, N, 2E]
            node_updates = self.node_mlp(node_inputs)  # [B, N, E]
            
            return node_updates, edge_updates
        
    def forward_sparse(self, x, edge_index, edge_attr):
        """Standard implementation using sparse operations"""
        # Extract source and target nodes
        src, dst = edge_index
        x_src = x[src]
        x_dst = x[dst]
        
        # Compute edge updates (equation 11)
        edge_update = self.edge_mlp(torch.cat([edge_attr, x_src, x_dst], dim=-1))
        
        # Aggregate using PyG's scatter operations
        aggregated = torch.zeros_like(x)
        for i in range(x.size(1)):
            aggregated[:, i] = torch.scatter_add(
                aggregated[:, i], 
                0, 
                dst, 
                edge_update[:, i]
            )
        
        # Compute node updates (equation 12)
        node_update = self.node_mlp(torch.cat([x, aggregated], dim=1))
        
        return node_update, edge_update
        
    def forward(self, x, edge_index=None, edge_attr=None, use_dense=False):
        """
        Unified forward method for TorchScript compatibility.
        
        Args:
            x: Node features [B*N, E] or [B, N, E]
            edge_index: Edge indices for sparse mode
            edge_attr: Edge features for sparse mode
            use_dense: Whether to use dense operations
        """
        if use_dense:
            # Handle 2D input format by estimating the reshape parameters
            if x.dim() == 2:
                # Estimate N from the total number of nodes
                N = int(x.size(0) ** 0.5)
                B = x.size(0) // N
                reshaped_x = x.view(B, N, -1)
                node_update, _ = self.forward_dense(reshaped_x)
                return node_update.view(B*N, -1), None
            else:
                # Input is already in [B, N, E] format
                node_update, _ = self.forward_dense(x)
                return node_update, None
        else:
            # Use sparse implementation
            return self.forward_sparse(x, edge_index, edge_attr)
