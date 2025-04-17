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
                 heights,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.channels_out = channels_out
        
        self.max_skip = max_skip
        self.fully_connected = fully_connected
        self.edge_channels_in = edge_channels_in
        
        self.heights = heights
        
        channels_in = channel_3d # + channel_2d

        self.encoder = Encoder(channels_in, edge_channels_in, embed_dim, emb_dropout)
        self.processor = Processor(embed_dim, depth=depth, dropout=dropout)
        self.decoder = Decoder(embed_dim, channels_out, dropout=dropout)
        
        self.sigmoid = nn.Sigmoid()
    


    def forward(self, x3d_norm, x2d_norm, x2d_orig):
        B, L, _ = x3d_norm.shape

        # encode the height and surface data
        x = torch.cat([x2d_norm.unsqueeze(1), x3d_norm], dim=1)
        
        # create the edge indices based on chosen connectivity pattern
        if self.fully_connected:
            edge_index = create_edge_index_fully_connected(L+1, B, x.device)
        else:
            edge_index = create_edge_index_multiMesh(L+1, B, x.device, self.max_skip)
        
        if self.heights is not None:
            edge_attr = create_height_edge_features(edge_index, self.heights, L+1, B, x.device)
            x, edge_attr_encoded = self.encoder(x, edge_attr)
        
        # If we still don't have edge attributes after encoding (edge_attr was None and no heights),
        # create default zero edge features
        else:  
            # Encode the node features first
            x = self.encoder.node_mlp(x)
            B, N, E = x.shape
            x = x.view(B*N, E)
            
            # Now create edge features with the SAME dimension as node features
            num_edges = edge_index.size(1)
            embed_dim = x.size(1)  # This is your embedding dimension
            edge_attr_encoded = torch.zeros(num_edges, embed_dim, device=x.device)
            
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
        # Initialize edge attributes if None
        if edge_attr is None:
            num_edges = edge_index.size(1)
            # Initialize with zeros matching the node embedding dimension
            edge_attr = torch.zeros(num_edges, x.size(1), device=x.device)
            
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


def create_height_edge_features(edge_index, heights, num_nodes, batch_size, device):
    """
    Create edge features based on height differences between connected nodes.
    
    Args:
        edge_index (torch.LongTensor): Edge index tensor with shape [2, num_edges]
        heights (torch.Tensor): Tensor of heights with shape [num_levels]
                               heights[0] = top of atmosphere (~65km)
                               heights[-1] = closest to surface (~20m)
        num_nodes (int): Number of nodes per batch (typically 71: 70 atm levels + surface)
        batch_size (int): Number of batches
        device (torch.device): Device to create tensors on
        
    Returns:
        edge_attr (torch.Tensor): Edge features based on height differences
    """
    # Extract source and target nodes for each edge
    src, dst = edge_index
    
    # Calculate node indices within each batch
    src_node = src % num_nodes    # Node index for source node
    dst_node = dst % num_nodes    # Node index for target node
    
    # Create a tensor to hold the height differences
    edge_features = torch.zeros(edge_index.shape[1], 1, device=device)
    
    # Define the surface node index (assuming it's the last node)
    surface_idx = heights.shape[0]  # If heights has 70 entries, surface_idx = 70
    
    # Create masks for different edge types
    src_is_atm = src_node < surface_idx  # Source is atmospheric level
    dst_is_atm = dst_node < surface_idx  # Destination is atmospheric level
    
    # Case 1: Both nodes are atmospheric levels
    both_atm = src_is_atm & dst_is_atm
    if both_atm.any():
        # For these edges, use the direct height differences
        # We're safe to index into heights for both nodes
        edge_features[both_atm, 0] = heights[dst_node[both_atm]] - heights[src_node[both_atm]]
    
    # Case 2: Source is atmospheric, destination is surface
    atm_to_surface = src_is_atm & ~dst_is_atm
    if atm_to_surface.any():
        # Height difference is negative height of source (going down to surface at 0)
        edge_features[atm_to_surface, 0] = -heights[src_node[atm_to_surface]]
    
    # Case 3: Source is surface, destination is atmospheric
    surface_to_atm = ~src_is_atm & dst_is_atm
    if surface_to_atm.any():
        # Height difference is positive height of destination (going up from surface)
        edge_features[surface_to_atm, 0] = heights[dst_node[surface_to_atm]]
        
    # Normalize by the maximum height (top of atmosphere)
    edge_features = edge_features / heights[0]
    
    return edge_features


# Add a function to load heights from NetCDF files
def load_vertical_heights(filename):
    """
    Load vertical height levels from a NetCDF file.
    
    Args:
        filename (str): Path to the NetCDF file containing heights
        
    Returns:
        heights (torch.Tensor): Tensor of heights
    """
    try:
        import xarray as xr
        
        # Open the NetCDF file
        ds = xr.open_dataset(filename)
        
        # Check if z_ifc exists in the dataset
        if 'z_ifc' in ds.data_vars:
            # Extract the first column of heights
            if len(ds['z_ifc'].shape) >= 3:
                heights = ds['z_ifc'][:,0,0].values  # First column
            else:
                heights = ds['z_ifc'][:].values
                
            # Convert to PyTorch tensor
            heights_tensor = torch.tensor(heights, dtype=torch.float32)
            
            # Close the dataset
            ds.close()
            
            return heights_tensor
        else:
            print("Warning: 'z_ifc' variable not found in the NetCDF file.")
            ds.close()
            return None
            
    except Exception as e:
        print(f"Error loading heights from {filename}: {e}")
        return None


