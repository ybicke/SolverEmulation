import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from utils.graph_3d import get_3d_graph

class GNN3D(nn.Module):
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
        
        # Store graph parameters for later use
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

    def forward(self, x3d_norm, x2d_norm):
        """
        Forward pass for the 3D GNN Tendency model.
        
        Args:
            x3d_norm: Normalized 3D input data with interpolated vertical wind already concatenated
            x2d_norm: Normalized 2D input data
        
        Returns:
            Model predictions
        """
        # Feature preparation
        B, N, L, _ = x3d_norm.shape
        
        # Process 2D features - broadcast to all vertical levels
        features_2d = x2d_norm.unsqueeze(2)
        repeat_2d_at_all_levels = features_2d.repeat(1, 1, L, 1) 
        
        # Combine 3D and 2D features
        x = torch.cat([x3d_norm, repeat_2d_at_all_levels], dim=-1)
        
        # Graph construction or retrieval
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
        
        # Edge feature initialization - start with zeros
        num_edges = batch_edge_index.size(1)
        edge_attr = torch.zeros(num_edges, self.embed_dim, device=x.device)
        #edge_attr_encoded = self.encoder.edge_mlp(edge_attr)
        
        # Node encoding - flatten for processing
        x_features = x.reshape(B * N * L, -1)
        x_encoded = self.encoder.node_mlp(x_features)
        
        # Message passing through GNN layers
        x_processed = self.processor(x_encoded, batch_edge_index, edge_attr)


        # Decoding and output
        x_decoded = self.decoder(x_processed)
        
        # Reshape back to batch format
        x_output = x_decoded.view(B, N, L, self.channels_out)
        
        return x_output


class Encoder(nn.Module):
    """
    Encodes node and edge features.
    
    Args:
        channels_in (int): Number of input channels for node features
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
            x (torch.Tensor): Node features [B*N*(L+1), channels_in]
            edge_attr (torch.Tensor): Edge features [num_edges, edge_channels_in] or None
            
        Returns:
            tuple: (encoded_node_features, encoded_edge_features)
        """
        x = self.node_mlp(x)
        
        if self.edge_mlp is not None and edge_attr is not None:
            edge_attr = self.edge_mlp(edge_attr)
            
        return x, edge_attr


class GNNLayer(MessagePassing):
    """
    Graph Neural Network layer for message passing with attention to both node and edge features.
    
    Args:
        embed_dim (int): Dimension of node and edge embeddings
        dropout (float): Dropout probability
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
        Forward pass for the GNN layer.
        
        Args:
            x (torch.Tensor): Node features [num_nodes, embed_dim]
            edge_index (torch.Tensor): Edge indices [2, num_edges]
            edge_attr (torch.Tensor): Edge features [num_edges, embed_dim]
            
        Returns:
            tuple: (node_updates, edge_updates)
        """
        # Extract source and target nodes for each edge
        src, dst = edge_index
        x_src = x[src]
        x_dst = x[dst]
        
        # Compute edge updates
        edge_update = self.edge_mlp(torch.cat([edge_attr, x_src, x_dst], dim=-1))
        
        # Aggregate messages at nodes
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
    Processes the graph using message passing layers.
    
    Args:
        embed_dim (int): Dimension of node and edge embeddings
        depth (int): Number of GNN layers
        dropout (float): Dropout probability
    """
    def __init__(self, embed_dim, depth, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            GNNLayer(embed_dim=embed_dim, dropout=dropout)
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