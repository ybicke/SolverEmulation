import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data
from copy import deepcopy

from einops import rearrange
import matplotlib.pyplot as plt

from torch_geometric.nn import GATConv

_logger = logging.getLogger(__name__)

#################################
# Reuse the Normalization Class #
#################################

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
    
    

class GATResidualLayer(nn.Module):
    """ A single GNN layer that applies GAT convolution + residual connection. """
    def __init__(self, embed_dim, heads=4, dropout=0.0):
        super().__init__()
        self.gat = GATConv(in_channels=embed_dim,
                           out_channels=embed_dim // heads,
                           heads=heads,
                           dropout=dropout,
                           add_self_loops=True)
        self.lin = nn.Linear(embed_dim, embed_dim)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(embed_dim)  # optional layer norm

    def forward(self, x, edge_index):
        x_res = x

        # GAT convolution
        x_mp = self.gat(x, edge_index)
        # Project back to embed_dim (in case heads > 1)
        x_mp = self.lin(x_mp)
        x = x_mp + x_res

        x = self.norm(x)
        x = self.act(x)

        return x

class GNNColumnNet(nn.Module):
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
                 heads=4,
                 mean2d=None,
                 var2d=None,
                 mean3d=None,
                 var3d=None,
                 *args,
                 **kwargs):
        super().__init__()

        self.heads = heads
        self.channels_out = channels_out
        self.swflx_idx = swflx_idx
        self.lwflx_idx = lwflx_idx
        self.cosmu0_idx = cosmu0_idx
        self.tsfctrad_idx = tsfctrad_idx

        self.normalizer2d = Normalization(std=torch.sqrt(var2d), mean=mean2d)
        self.normalizer3d = Normalization(std=torch.sqrt(var3d), mean=mean3d)

        self.node_encoder_3d = nn.Sequential(
            nn.Linear(channels_in, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(emb_dropout)
        )

        self.node_encoder_2d = nn.Sequential(
            nn.Linear(channels_in, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(emb_dropout)
        )

        self.input_drop = nn.Dropout(dropout)

        self.gnn_processor = nn.ModuleList([
            GATResidualLayer(embed_dim=embed_dim, heads=self.heads, dropout=dropout)
            for _ in range(depth)
        ])

        self.output_norm = nn.LayerNorm(embed_dim)

        self.output_decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, channels_out),
            nn.Sigmoid()
        )

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
    
    
    
    def create_edge_index(self, L, B, device):
        # Create edge indices for a single column
        single_column_edge_index = torch.stack([torch.arange(L-1), torch.arange(1, L)], dim=0)

        # Create a batch index to offset the node indices for each column
        batch_index = torch.arange(B).unsqueeze(1).repeat(1, L-1).view(-1)

        # Offset the node indices for each column in the batch
        batch_edge_index = single_column_edge_index.repeat(1, B) + batch_index * L
        batch_edge_index = batch_edge_index.to(device)

        return batch_edge_index

    def forward(self, x3d, x2d):
        B, L, _ = x3d.shape

        x3d = self.normalizer3d(x3d)
        x2d = self.normalizer2d(x2d)

        node_3d = self.node_encoder_3d(x3d)
        node_2d = self.node_encoder_2d(x2d)

        x = torch.cat([node_3d, node_2d.unsqueeze(1)], dim=1)
        x = self.input_drop(x)

        edge_index = self.create_edge_index(L+1, B, x.device)

        x = x.view(B*(L+1), -1)
        for layer in self.gnn_processor:
            x = layer(x, edge_index)

        x = x.view(B, L+1, -1)
        x = self.output_norm(x)

        x = self.output_decoder(x)
        x = self._scale_output(x, x2d)

        return x.squeeze()