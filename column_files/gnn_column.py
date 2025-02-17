import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from einops import rearrange
import matplotlib.pyplot as plt

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


###################################
# Simple Graph Aggregation Layer  #
###################################

#  ---------------------------------------------------------------------------------
#  | Similarities to Transformer (AFNO) blocks:                                     |
#  | - Both attempt to mix/transform intermediate representations of each "token"   |
#  |   (or "node" in GNN) via a learned function (linear layers + activations).     |
#  | - Both produce updated embeddings that feed into subsequent layers.           |
#  |--------------------------------------------------------------------------------|
#  | Differences from Transformer (AFNO) blocks:                                   |
#  | - GNN uses local neighbor aggregation (explicit adjacency in the chain),       |
#  |   whereas AFNO uses a global spectral mixing layer.                            |
#  | - There's no attention mechanism here; the aggregator is a simple average      |
#  |   of neighbors rather than a weighted attention across all positions.          |
#  ---------------------------------------------------------------------------------

class GraphLayer(nn.Module):
    """
    A simple 1D chain-based GNN layer that aggregates neighbor features.
    For each node (layer), we average neighbor + self features, then
    apply a feedforward + nonlinearity.

    """
    def __init__(self, embed_dim):
        super().__init__()
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        """
        x shape: (B, L, E)
        For Option B, we have L = 70 + 1 nodes now:
            - 70 atmospheric layers
            - 1 extra node for the surface

        We'll handle neighbor aggregation by manually checking boundary conditions,
        so the surface node (index = L-1) only neighbors the bottom layer (index = L-2).
        """
        B, L, E = x.shape
        # Create neighbor-averaged features (simple aggregator).
        # For each layer i, gather features from i, i-1, i+1 if in range.
        neighbor_sum = torch.zeros_like(x)
        
        # For convenience, pad left and right to handle boundary indexing
        # shape after pad: (B, L+2, E)
        x_padded = F.pad(x, (0, 0, 1, 1), mode='reflect')

        for i in range(L):
            neighbor_sum[:, i] = x_padded[:, i] + x_padded[:, i+1] + x_padded[:, i+2]
        
        neighbor_avg = neighbor_sum / 3.0
        out = self.fc(neighbor_avg)
        out = self.act(out)
        return out


###############################
# Main GNN Model for Columns #
###############################

class GNNColumnNet(nn.Module):
    """
    A simple GNN-based model for 1D atmospheric columns.

    ---------------------------------------------------------------------------------
    | Similarities to Transformer (AFNO) architecture:                              |
    |                                                                               |
    | 1) Input Normalization and Embedding:                                         |
    |    Just like in AFNO (which normalizes inputs and applies a patch embedding), |
    |    we normalize inputs here and linearly project them into an embedding       |
    |    dimension (see node_projection_3d and node_projection_2d).                 |
    |                                                                               |
    | 2) Stacked Layers:                                                            |
    |    AFNO has multiple Block modules (each with spectral mixing + MLP). GNN has |
    |    multiple GraphLayer modules (each with local neighbor aggregation + MLP).  |
    |                                                                               |
    | 3) Final Head + Sigmoid:                                                      |
    |    Both produce channel outputs (channels_out) and then apply a sigmoid       |
    |    to keep predictions in [0,1] for flux tasks, followed by re-scaling        |
    |    (physical interpretability).                                               |
    |                                                                               |
    |--------------------------------------------------------------------------------
    | Differences from Transformer (AFNO) architecture:                             |
    |                                                                               |
    | 1) Local vs. Global Mixing:                                                   |
    |    - AFNO uses a Fourier/spectral approach to mix information across the      |
    |      entire spatial dimension globally.                                       |
    |    - The GNN node embeddings only exchange information among immediate        |
    |      neighbors in the vertical chain (or farther if you define more edges).   |
    |                                                                               |
    | 2) No Attention Mechanism:                                                    |
    |    - Transformers typically have multi-head self-attention.                   |
    |    - GNN here does neighbor averaging (GraphLayer) instead of attention.      |
    |                                                                               |
    | 3) Output Reshaping:                                                          |
    |    - AFNO expands the embedding to patch_size * channels_out, reshaping       |
    |      for 2D patches.                                                          |
    |    - GNN just outputs (B, L, channels_out), directly matching the 1D chain.   |
    |--------------------------------------------------------------------------------
    |
    | "depth" => number of GraphLayers (akin to number of Transformer blocks).
    |
    | This class reuses the same Normalization logic for 2D/3D inputs and a similar
    | final scaling procedure used by AFNO to unscale fluxes.
    |
    """
    def __init__(self,
                 patch_size,
                 num_cells,
                 embed_dim,
                 depth,
                 dropout,
                 emb_dropout=0.0,
                 channels_in=6,
                 channels_out=4,
                 height=71,
                 swflx_idx=[2, 3],
                 lwflx_idx=[0, 1],
                 cosmu0_idx=1,
                 tsfctrad_idx=5,
                 mean2d=None,
                 var2d=None,
                 mean3d=None,
                 var3d=None,
                 device=None,
                 uniform_drop=False,
                 drop_path_rate=0.,
                 mlp_ratio=4.0,
                 hard_thresholding_fraction=1.0,
                 sparsity_threshold=0.01,
                 *args,
                 **kwargs):
        super().__init__()

        self.height = height
        self.channels_out = channels_out
        self.swflx_idx = swflx_idx
        self.lwflx_idx = lwflx_idx
        self.cosmu0_idx = cosmu0_idx
        self.tsfctrad_idx = tsfctrad_idx

        # Normalizers (reused from AFNO approach)
        self.normalizer2d = Normalization(std=torch.sqrt(var2d), mean=mean2d)
        self.normalizer3d = Normalization(std=torch.sqrt(var3d), mean=mean3d)

        # Node embedding dimension
        self.embed_dim = embed_dim

        # Simple linear projection of 3D input (B, L, channels_in) -> (B, L, E)
        self.node_encoder_3d = nn.Sequential(
            nn.Linear(channels_in, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(emb_dropout)
        )

        # 2D input embedding (could be stacked or added to each node embedding)
        self.node_encoder_2d = nn.Sequential(
            nn.Linear(channels_in, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(emb_dropout)
        )

        # Dropout for node embeddings after combination
        self.input_drop = nn.Dropout(dropout)

        # Build several GNN layers
        self.gnn_processor = nn.ModuleList([
            GraphLayer(embed_dim) for _ in range(depth)
        ])

        self.output_norm = nn.LayerNorm(embed_dim)

        # You can match AFNO's MLP head logic:
        self.output_decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, channels_out),
        )

    ################################
    # Radiation task specific bits #
    ################################
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
        """
        Reuses scaling logic from original AFNO code. 
        y_pred shape: (B, L, channels_out)
        x2d shape: (B, channels_in)
        """
        B, L, C = y_pred.shape
        scaled = []

        # For each channel, apply the appropriate post-processing
        for i in range(C):
            f_pred = y_pred[..., i:i+1]  # (B, L, 1)
            if i in self.swflx_idx:
                # Expand cosmu0 to (B, L, 1)
                cosmu0 = x2d[..., self.cosmu0_idx].unsqueeze(1)
                cosmu0 = cosmu0.repeat(1, L).unsqueeze(-1)
                scaled.append(self._unscale_swflx(f_pred, cosmu0))
            elif i in self.lwflx_idx:
                # Expand tsfctrad to (B, L, 1)
                tsfctrad = x2d[..., self.tsfctrad_idx].unsqueeze(1)
                tsfctrad = tsfctrad.repeat(1, L).unsqueeze(-1)
                scaled.append(self._unscale_lwflx(f_pred, tsfctrad))
            else:
                scaled.append(f_pred)

        return torch.cat(scaled, dim=-1)

    ##################
    # Forward Pass   #
    ##################

    def forward(self, x3d, x2d):
        # 1) Normalize inputs
        x3d = self.normalizer3d(x3d)
        x2d = self.normalizer2d(x2d)

        # 2) Encode inputs
        node_3d = self.node_encoder_3d(x3d)  # (B, 70, E)
        node_2d = self.node_encoder_2d(x2d)  # (B, E)
        node_2d = node_2d.unsqueeze(1)     # (B, 1, E)

        # 3) Concatenate encoded features
        x = torch.cat([node_3d, node_2d], dim=1)  # (B, 71, E)
        x = self.input_drop(x)

        # 4) Process with GNN layers
        for gnn_layer in self.gnn_processor:
            x = gnn_layer(x)

        # 5) Normalize processed features
        x = self.output_norm(x)  # (B, 71, E)

        # 6) Decode outputs
        x = self.output_decoder(x)  # (B, 71, channels_out)

        # 7) Apply sigmoid activation
        x = torch.sigmoid(x)

        # 8) Rescale outputs for flux channels
        x = self._scale_output(x, x2d)

        return x.squeeze()