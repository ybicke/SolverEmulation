import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing



class Normalization(nn.Module):
    """ Normalize the input based on mean and std """    
    def __init__(self, std, mean, axis=None):
        super().__init__()
        self.std = std
        self.mean = mean
        self.axis = axis

    def forward(self, x):
        assert x.size(dim=-1) == self.std.size(dim=0), \
            f'Dimension mismatch ( {x.size(dim=-1)} != {self.std.size(dim=0)})'
        return (x - self.mean)/self.std
    

class AtmosphericColumnGNN(nn.Module):
    def __init__(self,
                 embed_dim,
                 depth,
                 dropout,
                 max_skip,
                 emb_dropout=0.0,
                 channels_in=12,
                 channels_out=4,
                 swflx_idx=[2, 3],
                 lwflx_idx=[0, 1],
                 cosmu0_idx=1,
                 tsfctrad_idx=5,
                 mean2d=None,
                 var2d=None,
                 mean3d=None,
                 var3d=None,
                 *args,
                 **kwargs):
        super().__init__()

        self.channels_out = channels_out
        self.swflx_idx = swflx_idx
        self.lwflx_idx = lwflx_idx
        self.cosmu0_idx = cosmu0_idx
        self.tsfctrad_idx = tsfctrad_idx
        
        self.max_skip = max_skip

        self.normalizer2d = Normalization(std=torch.sqrt(var2d), mean=mean2d)
        self.normalizer3d = Normalization(std=torch.sqrt(var3d), mean=mean3d)

        self.encoder = Encoder(channels_in, embed_dim, emb_dropout)
        self.processor = Processor(embed_dim, depth = depth, dropout = dropout)
        self.decoder = Decoder(embed_dim, channels_out)
        
        self.learnable_level = nn.Parameter(torch.randn(1, 1, channels_out))
        self.sigmoid = nn.Sigmoid()
    


    def forward(self, x3d, x2d):
        B, L, _ = x3d.shape

        x2d_org = x2d.clone()

        x3d = self.normalizer3d(x3d)
        x2d = self.normalizer2d(x2d)

        # Expand x2d to have the same spatial dimensions as x3d
        x2d_expanded = x2d.unsqueeze(1).expand(-1, L, -1)  # [B, L, channels_in_2d]

        # Concatenate along the feature dimension
        x = torch.cat([x2d_expanded, x3d], dim=-1)  # [B, L, channels_in_3d + channels_in_2d]

        # encode the height and surface data
        x = self.encoder(x)

        # create the edge indices for the 1D chain graph and process the data with the GNN layers
        edge_index = create_edge_index_hirarchical(L, B, x.device, self.max_skip)
        x = self.processor(x, edge_index)   

        # reshape the output to the original shape and decode the output variables
        x = x.view(B, L, -1)  # Now using L, not L+1
        x = self.decoder(x)

        x = torch.cat([x, self.learnable_level.expand(B, -1, -1)], dim=1)  # [B, 71, channels_out]

        # scale the output variables to the original range
        x = self.sigmoid(x)
        x = self._scale_output(x, x2d_org)

        return x.squeeze()
    
    

    # Radiation task specifics:
    def _unscale_swflx(self, swflx, cosmu0):
        return torch.where(
            cosmu0 >= torch.tensor(1e-4, dtype=torch.float32),
            swflx * (cosmu0 * 1400),
            0
        )

    def _unscale_lwflx(self, lwflx, tsfctrad):
        stefan_boltzmann_const = torch.tensor(5.670374419e-08, dtype=torch.float32)
        return torch.where(
            tsfctrad >= torch.tensor(1e-4, dtype=torch.float32),
            lwflx * torch.pow(tsfctrad, 4) * stefan_boltzmann_const,
            lwflx
        )

    def _scale_output(self, y_pred, x2d):
        y_pred_scaled = []

        for i in range(y_pred.shape[-1]):
            f_pred = y_pred[..., i:i+1]

            if i in self.swflx_idx:
                cosmu0 = x2d[..., self.cosmu0_idx]
                cosmu0 = torch.tile(
                    cosmu0[..., None, None], 
                    (1, 1, f_pred.shape[-2], 1)
                )
                y_pred_scaled.append(self._unscale_swflx(f_pred, cosmu0))
            elif i in self.lwflx_idx:
                tsfctrad = x2d[..., self.tsfctrad_idx]
                tsfctrad = torch.tile(
                    tsfctrad[..., None, None], 
                    (1, 1, f_pred.shape[-2], 1)
                )
                y_pred_scaled.append(self._unscale_lwflx(f_pred, tsfctrad))
            else:
                y_pred_scaled.append(f_pred)

        y_pred = torch.cat(y_pred_scaled, dim=-1)
        return y_pred
        
    


class Encoder(nn.Module):
    """
    Input shape: [B, L+1, channels_in].
    Output shape: [B*(L+1), embed_dim].
    """
    def __init__(self, channels_in , embed_dim, emb_dropout):
        super().__init__()
        self.dropout = nn.Dropout(emb_dropout)
        self.mlp = nn.Sequential(
            nn.Linear(channels_in, embed_dim),  
            nn.SiLU(),
            nn.Dropout(emb_dropout),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        x = self.mlp(x)
        
        # flatten allows the GNN to treat each node (each level in each batch) independently during message passing
        # GNN operations are typically designed to work on a list of nodes, where each node can be processed in parallel.
        B, N, E = x.shape
        x = x.view(B*N, E)
        return x
    
    

class Processor(nn.Module):
    """
    Applies multiple GNN layers (message passing) with skip connections.
    Input shape: [B*(L+1), embed_dim].
    Output shape (unreshaped): same [B*(L+1), embed_dim].
    """
    def __init__(self, embed_dim, depth=16, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            GNNLayer(embed_dim=embed_dim, dropout=dropout)
            for _ in range(depth)
        ])
        self.layer_norm = nn.LayerNorm(embed_dim) 
        
    def forward(self, x, edge_index):
        for layer in self.layers:
            
            # Residual connection with LayerNorm for deep stack of GNN layers.
            # Helps with vanishing gradient problem.
            x = x + self.layer_norm(layer(x, edge_index))
        return x
    

class Decoder(nn.Module):
    """
    Input shape (reshaped): [B, L+1, embed_dim].
    Output shape: [B, L+1, channels_out], 
    """
    def __init__(self, embed_dim, channels_out, dropout=0.0):
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
    
    
class GNNLayer(MessagePassing):
    """
    A single GNN layer for the Processor.
    
    - Performs message passing to aggregate neighbor features.
    - An MLP updates node features, with skip connections for stability.
    - Iteratively captures complex dependencies across the graph.

    Input: [B*(L+1), embed_dim]
    Output: [B*(L+1), embed_dim]
    """
    def __init__(self, embed_dim, dropout=0.0):
        super().__init__(aggr='add')
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim), 
        )
        
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim), 
        )

        
    def forward(self, x, edge_index):
        x_org = x.clone()
        # edge indices guide which nodes send and receive messages
        # aggregation is implicitly defined by the propagate function
        
        # propagate() handles message passing.  It calls message(), aggregate(), and update().
        aggregated_messages = self.propagate(edge_index, x=x)
        
        # Concatenate original node features with aggregated messages.
        x = torch.cat([x, aggregated_messages], dim=1)
        
        x = x_org + self.node_mlp(x) # without residual was slightly better
        return x
    
    def message(self, x_j):
        # x_j: Features of the *source* node (j) of each edge.
        # Apply the learned edge transformation.  This is crucial, even without explicit edge features.
        
        #TODO: could implement edge features here. Would probably need a residual connection then. 
        # The edge_mlp transforms the source node features (x_j), and that transformed representation is used directly in the message aggregation. There's nothing to "add back" to.
        return self.edge_mlp(x_j)
    
    def update(self, aggr_out):
        # The GraphCast paper doesn't use a custom update function.
        # The aggregation result is directly used in the concatenation.
        return aggr_out
    
    



def create_edge_index_hirarchical(num_nodes, batch_size, device, max_skip):
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


