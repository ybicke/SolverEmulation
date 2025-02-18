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
        return edge_index



    def forward(self, x3d, x2d):
        B, L, _ = x3d.shape

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
        x = self._scale_output(x, x2d)

        return x.squeeze()
    
    



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
        x = self.dropout(x)
        
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

    def forward(self, x, edge_index):
        for layer in self.layers:
            # Skip connection
            x = x + layer(x, edge_index) 
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
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, channels_out),
        )

    def forward(self, x):
        B, N, E = x.shape
        x = x.view(B*N, E)
        x = self.mlp(x)
        x = x.view(B, N, )
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
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim), 
            # GraphCast suggested single linear layer. Need to check if this is correct.
        )

    def forward(self, x, edge_index):
        # Message passing
        x = self.propagate(edge_index, x=x)
        # MLP
        x = self.mlp(x)
        return x

    def message(self, x_j):
        return x_j
    
    
    
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
        B, L, C = y_pred.shape
        scaled = []

        for i in range(C):
            f_pred = y_pred[..., i:i+1]  # (B, L, 1)
            if i in self.swflx_idx:
                cosmu0 = x2d[..., self.cosmu0_idx].unsqueeze(1).repeat(1, L).unsqueeze(-1)
                scaled.append(self._unscale_swflx(f_pred, cosmu0))
            elif i in self.lwflx_idx:
                tsfctrad = x2d[..., self.tsfctrad_idx].unsqueeze(1).repeat(1, L).unsqueeze(-1)
                scaled.append(self._unscale_lwflx(f_pred, tsfctrad))
            else:
                scaled.append(f_pred)

        return torch.cat(scaled, dim=-1)
