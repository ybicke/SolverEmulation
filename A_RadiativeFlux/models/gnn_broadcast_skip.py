import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from .base_methods import BaseRadiationModel



class AtmosphericColumnGNN(BaseRadiationModel):
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
                 connections_per_node=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.channels_out = channels_out
        
        self.max_skip = max_skip
        self.embed_dim = embed_dim
        self.fully_connected = fully_connected
        self.edge_channels_in = edge_channels_in
        self.connections_per_node = connections_per_node
                
        channels_in = channel_3d + channel_2d

        self.encoder = Encoder(channels_in, edge_channels_in, embed_dim, emb_dropout)
        self.processor = Processor(embed_dim, depth=depth, dropout=dropout)
        self.decoder = Decoder(embed_dim, channels_out, dropout=dropout)
        
        self.sigmoid = nn.Sigmoid()
        
        # Cache for edge indices by batch size
        self.edge_index_cache = {}
        
        # Initialize device attribute
        self.device = kwargs.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    


    def _get_edge_index(self, num_nodes, batch_size):
        """
        Get the edge index from cache or create it if it doesn't exist.
        Using the batch size as a key to cache different edge indices.
        """
        # Check if we have already created this edge pattern for this batch size
        cache_key = batch_size
        
        if cache_key in self.edge_index_cache:
            return self.edge_index_cache[cache_key]
        
        # Create the edge indices based on chosen connectivity pattern
        if self.fully_connected:
            edge_index = create_edge_index_fully_connected(num_nodes, batch_size, self.device)
        elif self.connections_per_node is not None:
            edge_index = create_edge_index_optimal_sparse(num_nodes, batch_size, self.device, self.connections_per_node)
        else:
            edge_index = create_edge_index_multiMesh(num_nodes, batch_size, self.device, self.max_skip)
        
        # Cache for future use
        self.edge_index_cache[cache_key] = edge_index
        
        return edge_index


    def forward(self, x3d_norm, x2d_norm, x2d_orig):
        B, L, _ = x3d_norm.shape

        # Append surface features to each atmospheric level
        surface_features = x2d_norm.unsqueeze(1)
        repeat_surface_at_all_levels = surface_features.repeat(1, L, 1)
        augmented_atmospheric_column = torch.cat([x3d_norm, repeat_surface_at_all_levels], dim=-1)
        
        # Create extra surface nodes with zeros and match the batch dimension
        zero_surface_tensor = torch.zeros(B, 1, x2d_norm.shape[1], device=x3d_norm.device)
        append_zero_surface_tensor = torch.cat([zero_surface_tensor, surface_features], dim=-1)
        
        # Combine extra zero surface nodes with atmospheric columns
        x = torch.cat([augmented_atmospheric_column, append_zero_surface_tensor], dim=1)
        
        # Get the edge index, reusing from cache if available
        edge_index = self._get_edge_index(L+1, B)
        
        # Encode the node features first
        x = self.encoder.node_mlp(x)
        B, N, E = x.shape
        x = x.view(B*N, E)
        
        # Now create edge features with the SAME dimension as node features
        num_edges = edge_index.size(1)
        edge_attr_encoded = torch.zeros(num_edges, self.embed_dim, device=x.device)
        
        # Pass directly to processor, skipping edge encoding
        x = self.processor(x, edge_index, edge_attr_encoded)

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



def create_edge_index_optimal_sparse(num_nodes, batch_size, device, connections_per_node):
    """
    Creates an optimally distributed sparse connection pattern using mathematical principles.
    
    This method uses the golden ratio to create the most uniform distribution of connections
    possible, inspired by techniques used in low-discrepancy sampling and point distribution.
    
    The connections are bidirectional - for each connection from A→B, there's also a connection B→A.
    
    Args:
        num_nodes (int): Number of nodes in a single chain.
        batch_size (int): Number of chains (batches).
        device (torch.device): The device to push edge_index onto.
        connections_per_node (int): Number of outgoing connections per node (total connections will be 2x).
        
    Returns:
        edge_index (torch.LongTensor of shape [2, E]):
            Edge indices with optimally distributed bidirectional connections.
    """
    if connections_per_node >= num_nodes - 1:
        # If we want too many connections, fall back to fully connected
        return create_edge_index_fully_connected(num_nodes, batch_size, device)
    
    if connections_per_node <= 0:
        # No connections requested
        return torch.zeros(2, 0, dtype=torch.long, device=device)
    
    # Use vectorized operations for efficiency
    all_sources = []
    all_targets = []
    
    # Phi is the golden ratio conjugate, which gives the most uniform distribution
    phi = (5**0.5 - 1) / 2  # ≈ 0.618033988749895
    
    for node_idx in range(num_nodes):
        # Generate optimally distributed offsets using golden ratio
        # This creates the most even distribution mathematically possible
        offsets = []
        for i in range(1, connections_per_node + 1):
            # Scale connections to span the entire range
            offset = int(round(i * phi * (num_nodes - 1))) % (num_nodes - 1)
            # Add 1 to avoid zero offset
            offset = offset + 1 if offset < num_nodes - 1 else 1
            offsets.append(offset)
        
        # Create wrap-around connections
        targets = [(node_idx + offset) % num_nodes for offset in offsets]
        
        # Remove any self-connections or duplicates
        targets = [t for t in targets if t != node_idx]
        targets = list(dict.fromkeys(targets))  # remove duplicates while preserving order
        
        # If we lost some connections due to duplicates or self-references,
        # add additional connections to compensate
        if len(targets) < connections_per_node:
            # Find nodes not yet connected
            missing = connections_per_node - len(targets)
            candidates = [i for i in range(num_nodes) if i != node_idx and i not in targets]
            
            if candidates:
                # Use the same golden ratio principle to select additional targets
                indices = [(int(j * phi * len(candidates))) % len(candidates) 
                           for j in range(missing)]
                additional = [candidates[idx] for idx in indices]
                targets.extend(additional)
        
        # If we somehow got too many connections, trim
        targets = targets[:connections_per_node]
        
        # Add to our edge lists
        all_sources.extend([node_idx] * len(targets))
        all_targets.extend(targets)
    
    # Convert to tensor format
    if all_sources:
        # Create forward edges
        forward_edges = torch.tensor([all_sources, all_targets], device=device)
        
        # Create reverse edges (making the graph bidirectional)
        reverse_edges = torch.tensor([all_targets, all_sources], device=device)
        
        # Combine forward and reverse edges
        edge_index = torch.cat([forward_edges, reverse_edges], dim=1)
        
        # Repeat for batch_size
        base_edges = edge_index.repeat(1, batch_size)
        
        # Offset node indices for each batch
        offset = torch.arange(batch_size, device=device) * num_nodes
        offset = offset.repeat_interleave(base_edges.shape[1] // batch_size)
        edge_index = base_edges + offset.view(1, -1)
        
        return edge_index
    else:
        # Fallback for edge case with no edges
        return torch.zeros(2, 0, dtype=torch.long, device=device)

