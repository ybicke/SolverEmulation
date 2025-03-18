import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from triangle_files.triangle_graph import TriangleGraph

class GraphCastTriangle(nn.Module):
    def __init__(self,
                 grid_file_path,
                 embed_dim,
                 depth,
                 dropout=0.0,
                 channels_in_3d=6,  # Number of 3D features
                 channels_in_2d=6,  # Number of 2D features
                 channels_out=4,
                 input_height=70,   # Number of input height levels
                 output_height=71,  # Number of output height levels
                 swflx_idx=[2, 3],
                 lwflx_idx=[0, 1],
                 cosmu0_idx=1,
                 tsfctrad_idx=5,
                 mean2d=None,
                 var2d=None,
                 mean3d=None,
                 var3d=None,
                 device='cuda',
                 *args,
                 **kwargs):
        super().__init__()
        
        # Store dimensions
        self.input_height = input_height
        self.output_height = output_height
        
        # Store radiation-specific indices
        self.swflx_idx = swflx_idx
        self.lwflx_idx = lwflx_idx
        self.cosmu0_idx = cosmu0_idx
        self.tsfctrad_idx = tsfctrad_idx
        
        self.channels_out = channels_out
        self.device = device
        
        # Create the base graph structure for a single triangle
        self.triangle_graph = TriangleGraph(grid_file_path, device=device)
        self.base_edge_index = self.triangle_graph.initialize()
        self.triangle_graph.close()
        
        # Store dimensions for later use
        self.num_nodes_per_triangle = self.base_edge_index.max() + 1
        
        # Normalization layers
        self.normalizer2d = Normalization(std=torch.sqrt(var2d), mean=mean2d)
        self.normalizer3d = Normalization(std=torch.sqrt(var3d), mean=mean3d)
        
        # Core GNN components
        total_channels = channels_in_3d + channels_in_2d
        self.encoder = TriangleEncoder(total_channels, embed_dim, dropout)
        self.processor = TriangleProcessor(embed_dim, depth=depth, dropout=dropout)
        self.decoder = TriangleDecoder(embed_dim, channels_out, output_height)
        
    def _create_batch_edge_index(self, batch_size):
        """
        Creates edge indices for multiple triangles while preserving
        the complex connection structure from triangle_graph.py
        """
        batch_edge_index = self.base_edge_index.repeat(1, batch_size)
        offsets = torch.arange(batch_size, device=self.device) * self.num_nodes_per_triangle
        offsets = offsets.repeat_interleave(self.base_edge_index.size(1))
        batch_edge_index = batch_edge_index + offsets.view(1, -1)
        return batch_edge_index
    
    def forward(self, x3d, x2d):
        """
        Args:
            x3d: [batch_size, num_columns, input_height, num_3d_features]
            x2d: [batch_size, num_columns, num_2d_features]
        Returns:
            output: [batch_size, num_columns, output_height, channels_out]
        """
        # Store original x2d for radiation scaling
        x2d_original = x2d.clone()
        
        batch_size = x3d.shape[0]
        num_columns = x3d.shape[1]
        
        # Normalize inputs
        x3d = self.normalizer3d(x3d)
        x2d = self.normalizer2d(x2d)
        
        # Reshape and combine features
        x3d_flat = x3d.view(batch_size, num_columns * self.input_height, -1)
        x2d_expanded = x2d.unsqueeze(2).expand(-1, -1, self.input_height, -1)
        x2d_flat = x2d_expanded.reshape(batch_size, num_columns * self.input_height, -1)
        
        # Concatenate features
        x = torch.cat([x3d_flat, x2d_flat], dim=-1)  # [B, N*H_in, F3d+F2d]
        
        # Create the batch edge index that keeps triangles separate
        batch_edge_index = self._create_batch_edge_index(batch_size)
        
        # Process through GNN
        x = self.encoder(x)
        x = self.processor(x, batch_edge_index)
        x = self.decoder(x)  # Now outputs [B*N*H_out, channels_out]
        
        # Reshape output to match target shape
        output = x.view(batch_size, num_columns, self.output_height, self.channels_out)
        
        # Apply radiation-specific scaling
        output = self._scale_output(output, x2d_original)
        
        return output
    
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
        
    
    
    
    

class TriangleEncoder(nn.Module):
    """
    Input shape: [B, N*H, channels_in] where:
        B: batch size (number of triangles)
        N: number of columns in triangle (4096)
        H: number of height levels (70)
    Output shape: [B*N*H, embed_dim]
    """
    def __init__(self, channels_in, embed_dim, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels_in, embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        # x shape: [B, N*H, channels_in]
        B, NH, C = x.shape
        # Flatten for PyG message passing
        x = x.reshape(B*NH, C)
        return self.mlp(x)
    
    

class TriangleProcessor(nn.Module):
    """
    Processes the graph using message passing layers.
    Uses the complex edge structure defined in triangle_graph.py
    """
    def __init__(self, embed_dim, depth=16, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            TriangleGNNLayer(embed_dim=embed_dim, dropout=dropout)
            for _ in range(depth)
        ])
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, edge_index):
        """
        Args:
            x: [B*N*H, embed_dim] - flattened node features
            edge_index: [2, num_edges] - includes both vertical and horizontal connections
                                       as defined in triangle_graph.py
        """
        for layer in self.layers:
            x = x + self.layer_norm(layer(x, edge_index))
        return x
    
    
    
class TriangleGNNLayer(MessagePassing):
    """
    Message passing layer that uses the complex edge structure.
    Handles both vertical connections between heights and 
    horizontal connections between neighboring columns.
    """
    def __init__(self, embed_dim, dropout=0.0):
        # Use 'add' aggregation as in triangle_graph.py
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
        """
        Uses the edge_index from triangle_graph.py which includes:
        - Vertical connections between height levels
        - Horizontal connections between neighboring columns
        - Properly filtered edges at triangle boundaries
        """
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        # Transform features of source nodes
        return self.edge_mlp(x_j)

    def update(self, aggr_out, x):
        # Update node features with aggregated messages
        return self.node_mlp(torch.cat([x, aggr_out], dim=1))
        
    def edge_update(self, aggr_out, edge_index, num_nodes):
        # Required by MessagePassing but not used in this implementation
        return aggr_out
    
    

class TriangleDecoder(nn.Module):
    """
    Input shape: [B*N*H_in, embed_dim]
    Output shape: [B*N*H_out, channels_out]
    
    Where:
        B: batch size (number of complete triangles)
        N: number of columns in triangle (4096)
        H_in: number of input height levels (70)
        H_out: number of output height levels (71)
    """
    def __init__(self, embed_dim, channels_out, output_height, dropout=0.0):
        super().__init__()
        self.channels_out = channels_out
        self.output_height = output_height
        
        # Process features
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, channels_out),
        )

    def forward(self, x):
        # x shape: [B*N*H_in, embed_dim]
        BNH, E = x.shape
        B = BNH // 70  # Number of batches
        
        # Process features
        x = self.mlp(x)  # [B*N*H_in, channels_out]
        
        # Reshape to [B, N, 70, channels_out]
        x = x.view(B, -1, 70, self.channels_out)
        
        # Add an extra height level by interpolating between the last two levels
        # This is more physically motivated than using a learnable parameter
        last_level = x[..., -1, :]  # [B, N, channels_out]
        second_last_level = x[..., -2, :]  # [B, N, channels_out]
        interpolated_level = 0.5 * (last_level + second_last_level)  # Simple linear interpolation
        
        # Concatenate the interpolated level
        x = torch.cat([x, interpolated_level.unsqueeze(2)], dim=2)  # [B, N, 71, channels_out]
        
        # Reshape back to expected format
        x = x.reshape(-1, self.channels_out)  # [B*N*71, channels_out]
        
        return x
    
    

class Normalization(nn.Module):
    """Normalize the input based on mean and std"""
    def __init__(self, std, mean):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, x):
        return (x - self.mean) / self.std