import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_methods import BaseRadiationModel
from .graph_3d_full import get_3d_graph
import math
from .data_utils import get_triangle_indices


class GenCastEncoder(nn.Module):
    """
    GenCast-style encoder that maps input features to latent representations.
    Uses graph message passing for encoding.
    """
    def __init__(self, input_dim, latent_dim, num_layers=2, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)
        
        # Multi-layer encoding with graph message passing
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=heads,
                dim_feedforward=latent_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            )
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        """
        x: [B, N*L, input_dim] - flattened input features
        """
        # Project to latent space
        x = self.input_proj(x)
        x = self.norm(x)
        
        # Multi-layer encoding
        for layer in self.encoder_layers:
            x = layer(x)
            
        return x


class GenCastDecoder(nn.Module):
    """
    GenCast-style decoder that maps latent representations back to output features.
    Uses graph message passing for decoding.
    """
    def __init__(self, latent_dim, output_dim, num_layers=2, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        
        # Multi-layer decoding with graph message passing
        self.decoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=heads,
                dim_feedforward=latent_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(latent_dim)
        self.output_proj = nn.Linear(latent_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        x: [B, N*L, latent_dim] - latent representations
        """
        # Multi-layer decoding
        for layer in self.decoder_layers:
            x = layer(x)
            
        # Project to output space
        x = self.norm(x)
        x = self.output_proj(x)
        x = self.sigmoid(x)
        
        return x


# Import the processor from your existing implementations
from .gencast_transformer_3d import NeighborhoodSelfAttention, GraphTransformerLayer
from .traditional_graph_transformer_3d import TraditionalNeighborhoodSelfAttention, TraditionalGraphTransformerLayer


class GenCastFullArchitecture(BaseRadiationModel):
    """
    Full GenCast-style architecture with proper Encoder -> Processor -> Decoder pipeline.
    
    Architecture:
    1. Encoder: Maps input features to latent space with graph message passing
    2. Processor: Graph transformer layers (hybrid or traditional)
    3. Decoder: Maps latent space back to output features with graph message passing
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
                 num_height_levels,
                 device,
                 division_factor,
                 heads=8,
                 dim_head=64,
                 mlp_ratio=4.0,
                 fully_connected=False,
                 disable_horizontal=False,
                 max_hops=2,
                 processor_type="hybrid",  # "hybrid" or "traditional"
                 encoder_layers=2,
                 decoder_layers=2,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.device = device
        self.channels_out = channels_out
        self.embed_dim = embed_dim
        self.num_height_levels = num_height_levels
        self.processor_type = processor_type
        
        # Store grid parameters
        self.grid_file_path = grid_file_path
        self.triangle_id = triangle_id
        self.division_factor = division_factor
        self.total_cols = total_cols
        self.fully_connected = fully_connected
        self.disable_horizontal = disable_horizontal
        
        # Input/Output dimensions
        total_input_channels = channels_in_3d + channels_in_2d
        
        # Get the actual number of columns we'll be processing
        triangle_indices = get_triangle_indices(triangle_id, division_factor, total_cols)
        actual_num_columns = len(triangle_indices)
        
        # Learnable position embeddings
        self.pos_embedding_3d = nn.Parameter(torch.randn(1, actual_num_columns * num_height_levels, embed_dim))
        
        # === ENCODER ===
        self.encoder = GenCastEncoder(
            input_dim=total_input_channels,
            latent_dim=embed_dim,
            num_layers=encoder_layers,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout
        )
        
        # === PROCESSOR === 
        mlp_dim = int(embed_dim * mlp_ratio)
        
        if processor_type == "hybrid":
            # Hybrid processor (your original approach)
            self.processor_layers = nn.ModuleList([
                GraphTransformerLayer(embed_dim, heads, dim_head, mlp_dim, dropout, max_hops)
                for _ in range(depth)
            ])
        elif processor_type == "traditional":
            # Traditional processor
            self.processor_layers = nn.ModuleList([
                TraditionalGraphTransformerLayer(embed_dim, heads, dim_head, mlp_dim, dropout, max_hops)
                for _ in range(depth)
            ])
        else:
            raise ValueError(f"Unknown processor_type: {processor_type}")
        
        # === DECODER ===
        self.decoder = GenCastDecoder(
            latent_dim=embed_dim,
            output_dim=channels_out,
            num_layers=decoder_layers,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout
        )
        
    def forward(self, x3d_norm, x2d_norm, x2d_orig):
        B, N, L, _ = x3d_norm.shape
        
        # Get graph structure
        edge_index, num_columns = get_3d_graph(
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
        
        # Debug information
        print(f"GenCast Full Architecture ({self.processor_type}) Input shape: B={B}, N={N}, L={L}")
        print(f"Edge index shape: {edge_index.shape}")
        print(f"Processor type: {self.processor_type}")
        
        # Prepare features
        x2d_repeated = x2d_norm.unsqueeze(2).repeat(1, 1, L, 1)
        x_concat = torch.cat([x3d_norm, x2d_repeated], dim=-1)
        
        # Flatten for processing
        x_flat = x_concat.view(B, N * L, -1)  # [B, N*L, total_input_channels]
        
        # === ENCODER ===
        print("Running encoder...")
        x_encoded = self.encoder(x_flat)  # [B, N*L, embed_dim]
        
        # Add position embeddings
        pos_emb_size = min(x_encoded.size(1), self.pos_embedding_3d.size(1))
        x_encoded[:, :pos_emb_size, :] = x_encoded[:, :pos_emb_size, :] + self.pos_embedding_3d[:, :pos_emb_size, :]
        
        # === PROCESSOR ===
        print(f"Running {self.processor_type} processor...")
        x_processed = x_encoded
        for layer in self.processor_layers:
            x_processed = layer(x_processed, edge_index, num_columns)
        
        # === DECODER ===
        print("Running decoder...")
        x_decoded = self.decoder(x_processed)  # [B, N*L, channels_out]
        
        # Reshape back to 3D structure
        x = x_decoded.view(B, N, L, self.channels_out)  # [B, N, L, channels_out]
        
        # Scale output
        output = self._scale_output(x, x2d_orig)
        
        return output


class TraditionalGraphTransformerLayer(nn.Module):
    """
    Traditional Graph Transformer layer (imported for full architecture).
    """
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0, max_hops=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TraditionalNeighborhoodSelfAttention(dim, heads, dim_head, dropout, max_hops)
        self.norm2 = nn.LayerNorm(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, edge_index, num_columns):
        x = x + self.attn(self.norm1(x), edge_index, num_columns)
        x = x + self.mlp(self.norm2(x))
        return x 