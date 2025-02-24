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
    

    

#########################################################
# Multi-scale adjacency creation
#########################################################
def create_multiscale_edge_index(num_nodes, batch_size, device, max_scale=3):
    """
    Creates multi-scale adjacency for a 1D chain.

    For each scale s in [0..max_scale], we connect (i -> i + 2^s).
    That means each node i links to i+1, i+2, i+4, i+8, ...,
    as long as (i + 2^s) < num_nodes.

    Args:
        num_nodes (int): Number of nodes in a single chain.
        batch_size (int): Number of chains (batches).
        device (torch.device): The device to push edge_index onto.
        max_scale (int): The largest exponent for 2^s jumps.

    Returns:
        edge_index (torch.LongTensor of shape [2, E]):
            Concatenated edges for each scale, repeated for each chain in the batch.
    """
    all_edges = []

    for s in range(max_scale + 1):
        step = 2 ** s
        if step >= num_nodes:
            break  # step too large, won't form valid edges
        # Create edges for a single chain
        base_edges = torch.stack([
            torch.arange(num_nodes - step, device=device),
            torch.arange(step, num_nodes, device=device)
        ], dim=0)  # shape [2, (num_nodes - step)]

        # Repeat for batch_size
        base_edges = base_edges.repeat(1, batch_size)  # shape [2, (num_nodes - step) * batch_size]

        # Offset the node indices for each chain in the batch
        offset = torch.arange(batch_size, device=device) * num_nodes
        offset = offset.repeat_interleave(num_nodes - step)  # shape [(num_nodes - step) * batch_size]
        base_edges += offset.view(1, -1)
        all_edges.append(base_edges)

    # Concatenate edges from all scales => shape [2, total_edges]
    edge_index = torch.cat(all_edges, dim=1)
    return edge_index


#########################################################
# Example GNN building blocks
#########################################################
class Encoder(nn.Module):
    """
    Encodes raw node features into a higher-dimensional embedding.
    """
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, embed_dim),
            nn.SiLU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        # x: [B, N, in_channels]
        B, N, C = x.shape
        x = x.reshape(B*N, C)
        x = self.mlp(x)
        return x  # shape [B*N, embed_dim]


class HierarchicalProcessor(nn.Module):
    """
    Applies multiple GNN layers (message passing) 
    using the multi-scale adjacency from create_multiscale_edge_index.
    """
    def __init__(self, embed_dim, depth=8, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            GNNLayer(embed_dim, dropout=dropout) for _ in range(depth)
        ])

    def forward(self, x, edge_index):
        # x shape: [B*N, embed_dim]
        for layer in self.layers:
            # Residual skip => x_new = x + GNN(...)
            x = x + layer(x, edge_index)
        return x


class Decoder(nn.Module):
    """
    Decodes embeddings back to target dimension.
    """
    def __init__(self, embed_dim, out_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, out_channels)
        )

    def forward(self, x, B, N):
        # x shape: [B*N, embed_dim]
        x = self.mlp(x)
        x = x.view(B, N, -1)
        return x  # shape [B, N, out_channels]


class GNNLayer(MessagePassing):
    """
    A single GNN layer using PyTorch Geometric's MessagePassing.
    - 'aggr=add' means we sum messages from neighbors.
    """
    def __init__(self, embed_dim, dropout=0.0):
        super().__init__(aggr='add')
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x, edge_index):
        # x has shape [B*N, embed_dim]
        # edge_index has shape [2, E]
        out = self.propagate(edge_index, x=x)
        out = self.mlp(out)
        return out

    def message(self, x_j):
        # x_j are neighbor node features => we just pass them as is
        return x_j



#########################################################
# Putting it all together: 1D chain “GraphCast-like” GNN
#########################################################
class ChainGraphCast1D(nn.Module):
    """
    Example of a 1D chain GNN that mimics GraphCast's multi-scale design.

    Steps:
      1. Encodes node features (per-level or per-time-step).
      2. Builds multi-scale adjacency (Δ=1,2,4,...).
      3. Processor: multiple GNN layers see the big adjacency.
      4. Decoder: final projection to desired output dimension.
    """
    def __init__(self,
                 embed_dim,
                 depth,
                 dropout,
                 max_scale,
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
        
        self.embed_dim = embed_dim
        self.depth = depth
        self.max_scale = max_scale

        self.encoder = Encoder(channels_in, embed_dim)
        self.processor = HierarchicalProcessor(embed_dim, depth, dropout)
        self.decoder = Decoder(embed_dim, channels_out)
        
        self.sigmoid = nn.Sigmoid()



    def forward(self, x3d, x2d):
        """
        Args:
            x3d (torch.Tensor): shape [B, L, C]
              B = batch size,
              L = number of levels,
              C = in_channels (# features per node).
            x2d (torch.Tensor): shape [B, 1, C]
              B = batch size,
              1 = number of nodes in the 1D chain,
              C = in_channels (# features per node).
        Returns:
            out (torch.Tensor): shape [B, L+1, out_channels]
        """
        B, L, _ = x3d.shape

        x3d = self.normalizer3d(x3d)
        x2d = self.normalizer2d(x2d)
        x = torch.cat([x3d, x2d.unsqueeze(1)], dim=1)
        
        device = x.device

        # 1) Encode
        x = self.encoder(x)
        
        # 2) Build multi-scale adjacency
        edge_index = create_multiscale_edge_index(
            num_nodes=L+1,
            batch_size=B,
            device= device,
            max_scale=self.max_scale
        )

        # 3) Processor GNN
        x = self.processor(x, edge_index)

        # 4) Decoder
        x = self.decoder(x, B, L+1)
        
        # scale the output variables to the original range
        x = self.sigmoid(x)
        x = self._scale_output(x, x2d)

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
        