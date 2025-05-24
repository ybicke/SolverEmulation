import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

# from .base_methods import BaseRadiationModel

# Global cache for edge indices to avoid recomputing for identical parameters
_EDGE_INDEX_CACHE = {}

class AtmosphericColumnGNN(nn.Module):
    """
    Graph Neural Network for atmospheric column radiative transfer modeling.
    
    Creates a graph where nodes represent atmospheric levels, and edges represent
    connections between these levels. Edge features are derived from height differences
    when available.
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
                 device,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.channels_out = channels_out
        
        self.max_skip = max_skip
        self.embed_dim = embed_dim
        self.fully_connected = fully_connected
        self.edge_channels_in = edge_channels_in
        self.device = device
        channels_in = channel_3d + channel_2d

        self.encoder = Encoder(channels_in, edge_channels_in, embed_dim, emb_dropout)
        self.processor = Processor(embed_dim, depth=depth, dropout=dropout)
        self.decoder = Decoder(embed_dim, channels_out, dropout=dropout)
        
        self.sigmoid = nn.Sigmoid()
    


    def forward(self, x3d_norm, x2d_norm):
        # Ensure tensors are float32
        x3d_norm = x3d_norm.to(torch.float32)
        x2d_norm = x2d_norm.to(torch.float32)
        
        B, L, _ = x3d_norm.shape

        # Append surface features to each atmospheric level
        x2d_to_70 = x2d_norm.unsqueeze(1).repeat(1, x3d_norm.shape[1], 1)
        x = torch.cat((x3d_norm, x2d_to_70), dim=-1)
        
        # Get edge indices from cache or create new ones
        if self.fully_connected:
            edge_index = get_cached_edge_index_fully_connected(L, B, x.device)
        else:
            edge_index = get_cached_edge_index_multimesh(L, B, x.device, self.max_skip)
        
        # Encode the node features first
        x = self.encoder.node_mlp(x)
        _, _, E = x.shape
        x = x.view(B*L, E)
        
        # Now create edge features with the SAME dimension as node features
        num_edges = edge_index.size(1)
        edge_attr = torch.zeros(num_edges, self.embed_dim, device=x.device)
        #edge_attr_encoded = self.encoder.edge_mlp(edge_attr)

        
        # Pass directly to processor, skipping edge encoding
        x = self.processor(x, edge_index, edge_attr)

        # reshape the output to the original shape and decode the output variables
        x = x.view(B, L, -1)
        x = self.decoder(x)
        
        # scale the output variables to the original range
        # x = self.sigmoid(x)
        # x = self._scale_output(x, x2d_orig)

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
            GNNLayer(embed_dim=embed_dim, dropout=dropout)
            for _ in range(depth)
        ])
        
    def forward(self, x, edge_index, edge_attr):
            
        for layer in self.layers:
            # Compute node and edge updates
            node_update, edge_update = layer(x, edge_index, edge_attr)
            
            # Apply residual connections (equation 13)
            x = x + node_update
            edge_attr = edge_attr + edge_update
            
        # Return only the node features since that's all we need for the decoder
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

def get_cached_edge_index_fully_connected(num_nodes, batch_size, device):
    """
    Get a cached edge index for a fully connected graph, creating it if needed.
    
    Args:
        num_nodes: Number of nodes per graph
        batch_size: Batch size
        device: PyTorch device
        
    Returns:
        edge_index: Edge index tensor for the graph
    """
    # Create a unique key for this configuration
    cache_key = f"fully_connected_{num_nodes}_{batch_size}"
    
    # Check if we already have this configuration in the cache
    if cache_key in _EDGE_INDEX_CACHE:
        # Get from cache and ensure it's on the right device
        edge_index = _EDGE_INDEX_CACHE[cache_key]
        if edge_index.device != device:
            edge_index = edge_index.to(device)
        return edge_index
    
    # Not in cache, create it
    edge_index = create_edge_index_fully_connected(num_nodes, batch_size, device)
    
    # Store in cache (on CPU to save GPU memory)
    _EDGE_INDEX_CACHE[cache_key] = edge_index.cpu()
    
    return edge_index

def get_cached_edge_index_multimesh(num_nodes, batch_size, device, max_skip):
    """
    Get a cached edge index for a multimesh graph, creating it if needed.
    
    Args:
        num_nodes: Number of nodes per graph
        batch_size: Batch size
        device: PyTorch device
        max_skip: Maximum skip distance
        
    Returns:
        edge_index: Edge index tensor for the graph
    """
    # Create a unique key for this configuration
    cache_key = f"multimesh_{num_nodes}_{batch_size}_{max_skip}"
    
    # Check if we already have this configuration in the cache
    if cache_key in _EDGE_INDEX_CACHE:
        # Get from cache and ensure it's on the right device
        edge_index = _EDGE_INDEX_CACHE[cache_key]
        if edge_index.device != device:
            edge_index = edge_index.to(device)
        return edge_index
    
    # Not in cache, create it
    edge_index = create_edge_index_multiMesh(num_nodes, batch_size, device, max_skip)
    
    # Store in cache (on CPU to save GPU memory)
    _EDGE_INDEX_CACHE[cache_key] = edge_index.cpu()
    
    return edge_index

