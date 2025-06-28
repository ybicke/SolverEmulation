import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from utils.graph_3d import get_3d_graph



class Encoder(nn.Module):
    """
    Encodes node and edge features for 2D GraphCast-style processing.
    
    Args:
        channels_in (int): Number of input channels for node features (flattened 3D + 2D)
        edge_channels_in (int): Number of input channels for edge features
        embed_dim (int): Dimension of the embeddings
        dropout (float): Dropout probability
    """
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
        """
        Args:
            x (torch.Tensor): Node features [B*N, channels_in] (flattened columns)
            edge_attr (torch.Tensor): Edge features [num_edges, edge_channels_in] or None
            
        Returns:
            tuple: (encoded_node_features, encoded_edge_features)
        """
        x = self.node_mlp(x)
        
        if self.edge_mlp is not None and edge_attr is not None:
            edge_attr = self.edge_mlp(edge_attr)
            
        return x, edge_attr


class GNNLayer2D(MessagePassing):
    """
    2D GraphCast-style message passing layer for column-to-column communication.
    
    Follows the same architecture as gnn_3d.py but operates on 2D column graph.
    Each node represents a full atmospheric column with flattened height features.
    """
    def __init__(self, embed_dim, dropout):
        super().__init__(aggr='add')  # Use 'add' aggregation for messages
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * embed_dim, embed_dim),  # [edge_attr, x_src, x_dst]
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),  # [x, aggregated_messages]
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
        )
        
    def forward(self, x, edge_index, edge_attr):
        """        
        Args:
            x (torch.Tensor): Node features [B*N_columns, embed_dim] (flattened batch)
            edge_index (torch.Tensor): Edge indices [2, num_edges] (2D column edges)
            edge_attr (torch.Tensor): Edge features [num_edges, embed_dim]
            
        Returns:
            tuple: (node_updates, edge_updates)
        """
        src, dst = edge_index
        x_src = x[src]
        x_dst = x[dst]
        
        edge_update = self.edge_mlp(torch.cat([edge_attr, x_src, x_dst], dim=-1))
        
        aggregated_edge_updates = self.propagate(edge_index, x=x, edge_attr=edge_update)
        node_update = self.node_mlp(torch.cat([x, aggregated_edge_updates], dim=1))
        
        return node_update, edge_update
    
    def message(self, edge_attr):
        """
        Defines the messages passed along edges. Used internally by the MessagePassing class.
        """
        return edge_attr
    
    def update(self, aggr_out):
        """
        Updates node embeddings with aggregated messages. Used internally by the MessagePassing class.
        """
        return aggr_out


class Processor(nn.Module):
    """
    Processes the graph using 2D message passing layers.
    
    Args:
        embed_dim (int): Dimension of node and edge embeddings
        depth (int): Number of GNN layers
        dropout (float): Dropout probability
    """
    def __init__(self, embed_dim, depth, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            GNNLayer2D(embed_dim=embed_dim, dropout=dropout)
            for _ in range(depth)
        ])

    def forward(self, x, edge_index, edge_attr):
        for layer in self.layers:
            node_update, edge_update = layer(x, edge_index, edge_attr)
            
            # Apply residual connections
            x = x + node_update
            edge_attr = edge_attr + edge_update
                
        return x


class Decoder(nn.Module):
    """
    Decodes node features to output channels.
    
    Args:
        embed_dim (int): Dimension of node embeddings
        channels_out (int): Number of output channels
        dropout (float): Dropout probability
    """
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


def get_2d_edge_index(edge_index, num_columns, num_height_levels):
        """
        Extract 2D column-to-column connections from 3D edge index.
        
        Args:
            edge_index: [2, num_edges] 3D edge connectivity
            num_columns: number of columns in the grid
            num_height_levels: number of height levels
            
        Returns:
            column_edge_index: [2, num_2d_edges] column-to-column connections
        """
        # Convert 3D node indices to column indices
        src_nodes, dst_nodes = edge_index
        src_cols = src_nodes // num_height_levels
        dst_cols = dst_nodes // num_height_levels
        
    # Keep only edges between columns (horizontal connections)
        horizontal_mask = src_cols != dst_cols
        
        if horizontal_mask.sum() == 0:
            # No horizontal connections, return empty edge index
            return torch.zeros((2, 0), dtype=torch.long, device=edge_index.device)
        
        # Get unique column-to-column edges
        col_edges = torch.stack([src_cols[horizontal_mask], dst_cols[horizontal_mask]], dim=0)
        
        # Remove duplicates (since each 3D edge creates multiple column connections)
        unique_edges = torch.unique(col_edges, dim=1)
        
        return unique_edges


class GraphCastStyleGNN2D(nn.Module):
    """
    GraphCast-style 2D GNN for 3D atmospheric data on ICON triangular grid.
    
    Key features:
    1. Flattens all height levels into node features (no explicit vertical dimension)
    2. Pure 2D message passing on triangular grid between columns
    3. Follows the same architecture as gnn_3d.py but operates on 2D column graph
    4. Training pipeline compatibility
    
    This approach provides a GNN-based alternative to the GraphCast-style transformer
    by treating atmospheric columns as single nodes with rich vertical feature encoding.
    """
    def __init__(self,
                 total_cols,
                 grid_file_path,
                 triangle_id,
                 embed_dim,
                 depth,
                 dropout,
                 channels_in_3d,
                 channels_in_2d,
                 channels_out,
                 edge_channels_in,
                 num_height_levels,
                 device,
                 division_factor,
                 fully_connected,
                 disable_horizontal,
                 *args,
                 **kwargs):
        super().__init__()
        
        self.device = device
        self.channels_out = channels_out
        self.edge_channels_in = edge_channels_in
        self.embed_dim = embed_dim
        self.num_height_levels = num_height_levels
        
        # Store grid parameters for later use
        self.grid_file_path = grid_file_path
        self.triangle_id = triangle_id
        self.division_factor = division_factor
        self.total_cols = total_cols
        self.fully_connected = fully_connected
        self.disable_horizontal = disable_horizontal
        
        # Calculate flattened input size
        # All height levels are concatenated into a single feature vector per column
        total_3d_features = channels_in_3d * num_height_levels
        total_input_features = total_3d_features + channels_in_2d
        
        # Core GNN components (following gnn_3d.py architecture)
        self.encoder = Encoder(total_input_features, edge_channels_in, embed_dim, dropout)
        self.processor = Processor(embed_dim, depth=depth, dropout=dropout)
        
        # Output projection to flattened 3D output (like GenCast transformer)
        total_output_features = channels_out * num_height_levels
        self.decoder = Decoder(embed_dim, total_output_features, dropout)

    def forward(self, x3d_norm, x2d_norm):
        """
        Forward pass for the 2D GraphCast-style GNN model.
        
        Args:
            x3d_norm: [batch, num_columns, num_levels, channels_3d] normalized 3D data
            x2d_norm: [batch, num_columns, channels_2d] normalized 2D data
        
        Returns:
            output: [batch, num_columns, num_levels, channels_out] predictions
        """
        B, N, L, _ = x3d_norm.shape
        
        # Flatten vertical dimension into features (GraphCast-style)
        x3d_flat = x3d_norm.view(B, N, L * x3d_norm.size(-1))  # [B, N, L*C3d]
        
        # Concatenate 3D and 2D features
        # GraphCast-style approach treats 2D features as column-level context
        x = torch.cat([x3d_flat, x2d_norm], dim=-1)  # [B, N, L*C3d + C2d]
        
        # Get 3D graph structure for extracting 2D column connectivity
        batch_edge_index_3d, num_columns = get_3d_graph(
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
        
        # Extract 2D column-to-column edges from 3D graph
        col_edge_index = get_2d_edge_index(batch_edge_index_3d, num_columns, self.num_height_levels)
        
        # Create batched 2D edge index for all batch samples
        batch_edge_index_2d = []
        for b in range(B):
            batch_offset = b * num_columns
            batch_edges = col_edge_index + batch_offset
            batch_edge_index_2d.append(batch_edges)
        
        if batch_edge_index_2d:
            batch_edge_index_2d = torch.cat(batch_edge_index_2d, dim=1)
        else:
            batch_edge_index_2d = torch.zeros((2, 0), dtype=torch.long, device=x.device)
        
        # Edge feature initialization - start with zeros (already in embedding space)
        num_edges = batch_edge_index_2d.size(1)
        edge_attr = torch.zeros(num_edges, self.embed_dim, device=x.device)
        
        # Node encoding - flatten batch dimension for processing
        x_features = x.reshape(B * N, -1)  # [B*N, flattened_features]
        x_encoded = self.encoder.node_mlp(x_features)  # Only encode nodes, not edges
        
        # Message passing through GNN layers
        x_processed = self.processor(x_encoded, batch_edge_index_2d, edge_attr)
        
        # Decoding
        x_decoded = self.decoder(x_processed)
        
        # Reshape back to batch format and then to 3D structure
        x_output = x_decoded.view(B, N, -1)  # [B, N, L*channels_out]
        x_output_3d = x_output.view(B, N, L, self.channels_out)  # [B, N, L, channels_out]
        
        return x_output_3d


# Example usage and testing
if __name__ == "__main__":
    print("GraphCast-Style 2D GNN for ICON Triangular Grid")
    print("=" * 60)
    print("Key features:")
    print("1. Flattens all height levels into node features")
    print("2. Pure 2D message passing on triangular grid")
    print("3. Column-to-column message passing only")
    print("4. Follows gnn_3d.py architecture but on 2D column graph")
    print("5. Compatible with training pipeline interface")
    print() 