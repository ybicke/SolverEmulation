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
                 emb_dropout=0.0,
                 channels_in=6,
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

        self.normalizer2d = Normalization(std=torch.sqrt(var2d), mean=mean2d)
        self.normalizer3d = Normalization(std=torch.sqrt(var3d), mean=mean3d)

        self.encoder = Encoder(channels_in, embed_dim, emb_dropout)
        self.processor = Processor(embed_dim, depth = depth, dropout = dropout)
        self.decoder = Decoder(embed_dim, channels_out)
        
        self.sigmoid = nn.Sigmoid()



    def create_edge_index(self, num_nodes, batch_size, device):
        """
        Creates edge indices for a batch of 1D chain graphs.
        Each graph represents a linear chain where nodes are sequentially connected.
        Ensures unique node indices across graphs in the batch.

        Returns:
        - edge_index (torch.Tensor): A tensor of shape [2, num_edges * batch_size]
        containing the source and target node indices for all edges in the batch.
        """
        # Create edges for the 1D chain graph
        edge_index = torch.stack([
            torch.arange(num_nodes - 1, device=device),
            torch.arange(1, num_nodes, device=device)
        ], dim=0)

        # Repeat edge_index for each graph in the batch
        edge_index = edge_index.repeat(1, batch_size)

        # Offset the node indices for each graph in the batch
        batch_offset = (torch.arange(batch_size, device=device) * num_nodes)
        batch_offset = batch_offset.repeat_interleave(num_nodes - 1)

        # Optionally ensure edge_index is also explicitly on the device:
        edge_index = edge_index.to(device)
        edge_index += batch_offset.view(1, -1)

        # Create bidirectional edges:
        reverse_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)

        return edge_index



    def forward(self, x3d, x2d):
        B, L, _ = x3d.shape

        x2d_org = x2d.clone()

        x3d = self.normalizer3d(x3d)
        x2d = self.normalizer2d(x2d)

        # encode the height and surface data
        x = torch.cat([x3d, x2d.unsqueeze(1)], dim=1)
        x = self.encoder(x)
        
        # create the edge indices for the 1D chain graph and process the data with the GNN layers
        edge_index = self.create_edge_index(L+1, B, x.device)
        x = self.processor(x, edge_index)
        
        # reshape the output to the original shape and decode the output variables
        x = x.view(B, L+1, -1)
        x = self.decoder(x)
        
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
        # edge indices guide which nodes send and receive messages
        # aggregation is implicitly defined by the propagate function
        
        # propagate() handles message passing.  It calls message(), aggregate(), and update().
        aggregated_messages = self.propagate(edge_index, x=x)
        
        # Concatenate original node features with aggregated messages.
        x = torch.cat([x, aggregated_messages], dim=1)
        
        # MLP to update the node features. Residual connection proposed. Helps that the node doesn't forget its original features.
        # Without, the node_mlp would have to learn to both process the aggregated information and reconstruct the original information, which is a much harder task. 
        # Smoother Optimization Landscape: Residual connections create "shortcuts" in the optimization landscape. This makes it easier for the optimizer to find good solutions and avoids getting stuck in local minima.
        # the internal residual connection within each GNNLayer effectively makes each layer "deeper" in terms of its ability to learn complex transformations.
        # The GNNLayer-level residuals allow each GNNLayer to learn more complex transformations and preserve information.
        x = self.node_mlp(x)
        return x
    
    def message(self, x_j):
        # x_j: Features of the *source* node (j) of each edge.
        # Apply the learned edge transformation.  This is crucial, even without explicit edge features.
        return self.edge_mlp(x_j)
    
    def update(self, aggr_out):
        # The GraphCast paper doesn't use a custom update function.
        # The aggregation result is directly used in the concatenation.
        return aggr_out
    
    

