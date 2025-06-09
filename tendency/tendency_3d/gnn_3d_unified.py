import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from graph_3d_full import get_3d_graph
from .base_methods import BaseRadiationModel


class GNN3D(nn.Module):
    """Unified 3D GNN model that can handle different data types via mode parameter."""
    
    def __init__(self,
                 mode='tendency',  # 'tendency' or 'radiation'
                 total_cols=None,
                 grid_file_path=None,
                 triangle_id=None,
                 embed_dim=None,
                 depth=None,
                 dropout=None,
                 channels_in_3d=None,
                 channels_in_2d=None,
                 channels_out=None,
                 edge_channels_in=None,
                 num_height_levels=None,
                 device=None,
                 division_factor=None,
                 fully_connected=None,
                 disable_horizontal=None,
                 *args,
                 **kwargs):
        
        # Initialize appropriate parent class based on mode
        if mode == 'radiation':
            BaseRadiationModel.__init__(self, *args, **kwargs)
        super(GNN3D, self).__init__()
        
        self.mode = mode
        self.device = device
        self.channels_out = channels_out
        self.edge_channels_in = edge_channels_in
        self.embed_dim = embed_dim
        
        # Store graph parameters
        self.grid_file_path = grid_file_path
        self.triangle_id = triangle_id
        self.num_height_levels = num_height_levels
        self.division_factor = division_factor
        self.total_cols = total_cols
        self.fully_connected = fully_connected
        self.disable_horizontal = disable_horizontal
        
        # Core GNN components
        total_channels = channels_in_3d + channels_in_2d
        
        self.encoder = Encoder(total_channels, edge_channels_in, embed_dim, dropout)
        self.processor = Processor(embed_dim, depth=depth, dropout=dropout)
        self.decoder = Decoder(embed_dim, channels_out, dropout=dropout)
        
        # Mode-specific components
        if mode == 'radiation':
            self.sigmoid = nn.Sigmoid()

    def forward(self, x3d_norm, x2d_norm, x2d_orig=None):
        """
        Forward pass for the 3D GNN model.
        
        Args:
            x3d_norm: Normalized 3D input data
            x2d_norm: Normalized 2D input data
            x2d_orig: Original 2D data (required for radiation mode)
        
        Returns:
            Model predictions
        """
        # Feature preparation
        B, N, L, _ = x3d_norm.shape
        features_2d = x2d_norm.unsqueeze(2)
        repeat_2d_at_all_levels = features_2d.repeat(1, 1, L, 1)
        
        if self.mode == 'radiation':
            # Add extra "ones" level for radiation
            augmented_3d_column = torch.cat([x3d_norm, repeat_2d_at_all_levels], dim=-1)
            ones_2d = torch.ones(B, N, 1, augmented_3d_column.shape[-1], device=x3d_norm.device)
            x = torch.cat([ones_2d, augmented_3d_column], dim=2)
            L = L + 1  # Update L for radiation mode
        else:
            # Direct concatenation for tendency
            x = torch.cat([x3d_norm, repeat_2d_at_all_levels], dim=-1)
        
        # Graph construction
        batch_edge_index, N = get_3d_graph(
            grid_file_path=self.grid_file_path,
            triangle_id=self.triangle_id,
            num_height_levels=self.num_height_levels,
            batch_size=B,
            total_cols=self.total_cols,
            division_factor=self.division_factor,
            device=self.device,
            fully_connected=self.fully_connected,
            disable_horizontal=self.disable_horizontal
        )
        
        # Edge feature initialization
        num_edges = batch_edge_index.size(1)
        edge_attr = torch.zeros(num_edges, self.embed_dim, device=x.device)
        
        # Node encoding
        x_features = x.reshape(B * N * L, -1)
        x_encoded = self.encoder.node_mlp(x_features)
        
        # Message passing
        x_processed = self.processor(x_encoded, batch_edge_index, edge_attr)
        
        # Decoding and output
        x_decoded = self.decoder(x_processed)
        x_output = x_decoded.view(B, N, L, self.channels_out)
        
        # Mode-specific output processing
        if self.mode == 'radiation':
            x_output = self.sigmoid(x_output)
            return self._scale_output(x_output, x2d_orig)
        else:
            return x_output


# Keep the same Encoder, GNNLayer, Processor, and Decoder classes
class Encoder(nn.Module):
    """Encodes node and edge features."""
    def __init__(self, channels_in, edge_channels_in, embed_dim, dropout):
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(channels_in, embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
        )
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_channels_in, embed_dim),
            nn.SiLU(), 
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
        ) if edge_channels_in > 0 else None
        
    def forward(self, x, edge_attr):
        x = self.node_mlp(x)
        if self.edge_mlp is not None and edge_attr is not None:
            edge_attr = self.edge_mlp(edge_attr)
        return x, edge_attr


class GNNLayer(MessagePassing):
    """Graph Neural Network layer for message passing."""
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
        src, dst = edge_index
        x_src = x[src]
        x_dst = x[dst]
        
        edge_update = self.edge_mlp(torch.cat([edge_attr, x_src, x_dst], dim=-1))
        aggregated_edge_updates = self.propagate(edge_index, x=x, edge_attr=edge_update)
        node_update = self.node_mlp(torch.cat([x, aggregated_edge_updates], dim=1))
        
        return node_update, edge_update
    
    def message(self, edge_attr):
        return edge_attr
    
    def update(self, aggr_out):
        return aggr_out


class Processor(nn.Module):
    """Processes the graph using message passing layers."""
    def __init__(self, embed_dim, depth, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            GNNLayer(embed_dim=embed_dim, dropout=dropout)
            for _ in range(depth)
        ])

    def forward(self, x, edge_index, edge_attr):
        for layer in self.layers:
            node_update, edge_update = layer(x, edge_index, edge_attr)
            x = x + node_update
            edge_attr = edge_attr + edge_update
        return x


class Decoder(nn.Module):
    """Decodes node features to output channels."""
    def __init__(self, embed_dim, channels_out, dropout=0.0):
        super().__init__()
        self.channels_out = channels_out
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, channels_out),
        )

    def forward(self, x):
        return self.mlp(x)


# Convenience classes for backward compatibility
class GNN3dTendency(GNN3D):
    """GNN for tendency prediction."""
    def __init__(self, *args, **kwargs):
        super().__init__(mode='tendency', *args, **kwargs)


class GNN3dRadiation(GNN3D):
    """GNN for radiation modeling."""
    def __init__(self, *args, **kwargs):
        super().__init__(mode='radiation', *args, **kwargs) 