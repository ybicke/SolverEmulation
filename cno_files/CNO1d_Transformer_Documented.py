# CNO1d.py
# The CNO1d code has been modified from a tutorial featured in the 
# ETH Zurich course "AI in the Sciences and Engineering."
# Git page for this course: https://github.com/bogdanraonic3/AI_Science_Engineering 

# For up/downsampling, the antialias interpolation functions from the 
# torch library are utilized, limiting the ability to design
# your own low-pass filters at present.

# While acknowledging this suboptimal setup, the performance of CNO1d remains commendable. 
# Additionally, a training script is available, offering a solid foundation for personal projects.

import torch
import torch.nn as nn
import torch.nn.functional as F

# The CNO1d model implements a 1-dimensional Convolutional Neural Operator.
# It follows an encoder-decoder architecture with residual connections.

#---------------------
# Activation Function:
#---------------------

class CNO_LReLu(nn.Module):
    """
    Custom activation function for CNO1d.
    Applies upsampling, LeakyReLU activation, and downsampling.
    """
    def __init__(self, in_size, out_size):
        super(CNO_LReLu, self).__init__()
        self.in_size = in_size       # Input spatial size
        self.out_size = out_size     # Output spatial size
        self.act = nn.LeakyReLU()    # LeakyReLU activation function

    def forward(self, x):
        # Upsample the input by a factor of 2 using bicubic interpolation
        x = F.interpolate(
            x.unsqueeze(2),                # Add an extra dimension: [B, C, 1, L]
            size=(1, 2 * self.in_size),    # Target size after upsampling
            mode="bicubic",
            align_corners=False,
            antialias=True                 # Reduce aliasing artifacts
        )
        x = self.act(x)                    # Apply LeakyReLU activation
        # Downsample back to the original (or desired) size
        x = F.interpolate(
            x,
            size=(1, self.out_size),       # Target size after downsampling
            mode="bicubic",
            align_corners=False,
            antialias=True
        )
        return x[:, :, 0]                  # Remove the extra dimension and return output

#--------------------
# CNO Block:
#--------------------

class CNOBlock(nn.Module):
    """
    Basic building block of the CNO1d model.
    Consists of a convolution, optional batch normalization, and a custom activation function.
    """
    def __init__(self, in_channels, out_channels, in_size, out_size, use_bn=True):
        super(CNOBlock, self).__init__()

        self.in_channels = in_channels     # Number of input channels
        self.out_channels = out_channels   # Number of output channels
        self.in_size = in_size             # Input spatial size
        self.out_size = out_size           # Output spatial size

        # Convolutional layer with kernel size 3 and padding 1
        self.convolution = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            padding=1
        )

        # Optional batch normalization
        if use_bn:
            self.batch_norm = nn.BatchNorm1d(self.out_channels)
        else:
            self.batch_norm = nn.Identity()

        # Custom activation function that includes up/downsampling
        self.act = CNO_LReLu(
            in_size=self.in_size,
            out_size=self.out_size
        )

    def forward(self, x):
        x = self.convolution(x)    # Apply convolution
        x = self.batch_norm(x)     # Apply batch normalization
        x = self.act(x)            # Apply activation function with up/downsampling
        return x

#--------------------
# Lift/Project Block:
#--------------------

class LiftProjectBlock(nn.Module):
    """
    Lift block transforms input data to a higher-dimensional feature space.
    Project block brings data back to the output space.
    """
    def __init__(self, in_channels, out_channels, size, latent_dim=64):
        super(LiftProjectBlock, self).__init__()

        # Intermediate CNOBlock to increase feature dimensions
        self.inter_CNOBlock = CNOBlock(
            in_channels=in_channels,
            out_channels=latent_dim,
            in_size=size,
            out_size=size,
            use_bn=False
        )

        # Convolution to adjust the number of channels to the desired output
        self.convolution = nn.Conv1d(
            in_channels=latent_dim,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):
        x = self.inter_CNOBlock(x)   # Apply intermediate CNOBlock
        x = self.convolution(x)      # Apply convolution to adjust channels
        return x

#--------------------
# Residual Block:
#--------------------

class ResidualBlock(nn.Module):
    """
    Residual block consists of two convolutional layers with an activation in between.
    Implements a skip connection for residual learning.
    """
    def __init__(self, channels, size, use_bn=True):
        super(ResidualBlock, self).__init__()

        self.channels = channels     # Number of channels
        self.size = size             # Spatial size

        # First convolutional layer
        self.convolution1 = nn.Conv1d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=3,
            padding=1
        )

        # Optional batch normalization
        if use_bn:
            self.batch_norm1 = nn.BatchNorm1d(self.channels)
        else:
            self.batch_norm1 = nn.Identity()

        # Activation function
        self.act = CNO_LReLu(
            in_size=self.size,
            out_size=self.size
        )

        # Second convolutional layer
        self.convolution2 = nn.Conv1d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=3,
            padding=1
        )

        # Optional batch normalization
        if use_bn:
            self.batch_norm2 = nn.BatchNorm1d(self.channels)
        else:
            self.batch_norm2 = nn.Identity()

    def forward(self, x):
        out = self.convolution1(x)   # Apply first convolution
        out = self.batch_norm1(out)  # Apply batch normalization
        out = self.act(out)          # Apply activation function
        out = self.convolution2(out) # Apply second convolution
        out = self.batch_norm2(out)  # Apply batch normalization
        return x + out               # Add residual connection

#--------------------
# ResNet:
#--------------------

class ResNet(nn.Module):
    """
    ResNet consists of multiple ResidualBlocks applied sequentially.
    """
    def __init__(self, channels, size, num_blocks, use_bn=True):
        super(ResNet, self).__init__()

        self.channels = channels     # Number of channels
        self.size = size             # Spatial size
        self.num_blocks = num_blocks # Number of residual blocks

        # Create a sequential container of ResidualBlocks
        self.res_nets = nn.Sequential(*[
            ResidualBlock(
                channels=self.channels,
                size=self.size,
                use_bn=use_bn
            ) for _ in range(self.num_blocks)
        ])

    def forward(self, x):
        x = self.res_nets(x)         # Apply the sequence of ResidualBlocks
        return x

#--------------------
# Transformer Bottleneck:
#--------------------

class TransformerBottleneck(nn.Module):
    """
    Transformer encoder module used as a bottleneck to capture global dependencies.
    """
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerBottleneck, self).__init__()

        # Positional encoding to provide sequence order information
        self.positional_encoding = nn.Parameter(torch.zeros(1, d_model))

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        # x shape: [batch_size, channels, sequence_length]
        # Add positional encoding
        x = x + self.positional_encoding.unsqueeze(1)
        # Permute to [batch_size, sequence_length, channels]
        x = x.permute(0, 2, 1)
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        # Permute back to [batch_size, channels, sequence_length]
        x = x.permute(0, 2, 1)
        return x

#--------------------
# CNO1d Model:
#--------------------

class CNO1d(nn.Module):
    """
    CNO1d model implements a 1D Convolutional Neural Operator with an encoder-decoder architecture.
    Includes optional transformer bottleneck for capturing global dependencies.
    """
    def __init__(
        self,
        in_dim,                 # Number of input channels
        out_dim,                # Number of output channels
        size,                   # Spatial size of input/output
        N_layers,               # Number of encoder/decoder layers
        N_res=4,                # Number of residual blocks per level
        N_res_neck=4,           # Number of residual blocks in the bottleneck
        channel_multiplier=16,  # Multiplier for the number of channels
        use_bn=True,            # Use batch normalization
        transformer_d_model=None,       # Transformer model dimension
        transformer_nhead=None,         # Number of attention heads
        transformer_num_layers=None,    # Number of transformer layers
        transformer_dim_feedforward=2048,  # Feedforward network dimension
        transformer_dropout=0.1         # Dropout rate
    ):
        super(CNO1d, self).__init__()

        self.N_layers = int(N_layers)           # Number of encoder/decoder layers
        self.lift_dim = channel_multiplier // 2 # Dimension to lift input into
        self.in_dim = in_dim                    # Number of input channels
        self.out_dim = out_dim                  # Number of output channels
        self.channel_multiplier = channel_multiplier  # Channel multiplier

        #--------------------
        # Define channel sizes at each layer
        #--------------------
        # Encoder features: list of channels at each encoder layer
        self.encoder_features = [self.lift_dim]
        for i in range(self.N_layers):
            self.encoder_features.append(2 ** i * self.channel_multiplier)

        # Decoder features: reverse of encoder features
        self.decoder_features_in = self.encoder_features[1:]
        self.decoder_features_in.reverse()
        self.decoder_features_out = self.encoder_features[:-1]
        self.decoder_features_out.reverse()

        # Adjust decoder features to account for concatenation of skip connections
        for i in range(1, self.N_layers):
            self.decoder_features_in[i] = 2 * self.decoder_features_in[i]

        #--------------------
        # Define spatial sizes at each layer
        #--------------------
        self.encoder_sizes = []   # Spatial sizes in encoder
        self.decoder_sizes = []   # Spatial sizes in decoder
        for i in range(self.N_layers + 1):
            self.encoder_sizes.append(size // 2 ** i)
            self.decoder_sizes.append(size // 2 ** (self.N_layers - i))

        #--------------------
        # Define Lift and Project Blocks
        #--------------------
        # Lift Block: transforms input to higher-dimensional feature space
        self.lift = LiftProjectBlock(
            in_channels=self.in_dim,
            out_channels=self.encoder_features[0],
            size=size
        )

        # Project Block: transforms features back to output dimension
        self.project = LiftProjectBlock(
            in_channels=self.encoder_features[0] + self.decoder_features_out[-1],
            out_channels=self.out_dim,
            size=size
        )

        #--------------------
        # Define Encoder, ED Expansion, and Decoder networks
        #--------------------

        # Encoder: sequence of CNOBlocks that downsample and increase channels
        self.encoder = nn.ModuleList([
            CNOBlock(
                in_channels=self.encoder_features[i],
                out_channels=self.encoder_features[i + 1],
                in_size=self.encoder_sizes[i],
                out_size=self.encoder_sizes[i + 1],
                use_bn=use_bn
            ) for i in range(self.N_layers)
        ])

        # ED Expansion: adjusts skip connection features to match decoder dimensions
        self.ED_expansion = nn.ModuleList([
            CNOBlock(
                in_channels=self.encoder_features[i],
                out_channels=self.encoder_features[i],
                in_size=self.encoder_sizes[i],
                out_size=self.decoder_sizes[self.N_layers - i],
                use_bn=use_bn
            ) for i in range(self.N_layers + 1)
        ])

        # Decoder: sequence of CNOBlocks that upsample and decrease channels
        self.decoder = nn.ModuleList([
            CNOBlock(
                in_channels=self.decoder_features_in[i],
                out_channels=self.decoder_features_out[i],
                in_size=self.decoder_sizes[i],
                out_size=self.decoder_sizes[i + 1],
                use_bn=use_bn
            ) for i in range(self.N_layers)
        ])

        #--------------------
        # Define ResNets Blocks
        #--------------------

        # Residual Networks in the encoder
        self.res_nets = nn.ModuleList([
            ResNet(
                channels=self.encoder_features[l],
                size=self.encoder_sizes[l],
                num_blocks=N_res,
                use_bn=use_bn
            ) for l in range(self.N_layers)
        ])

        # Bottleneck Residual Network
        self.res_net_neck = ResNet(
            channels=self.encoder_features[self.N_layers],
            size=self.encoder_sizes[self.N_layers],
            num_blocks=N_res_neck,
            use_bn=use_bn
        )

        #--------------------
        # Define Transformer Bottleneck (Optional)
        #--------------------

        if (
            transformer_d_model is not None and
            transformer_nhead is not None and
            transformer_num_layers is not None
        ):
            # Ensure transformer_d_model matches bottleneck channels
            assert transformer_d_model == self.encoder_features[self.N_layers], \
                "transformer_d_model must match the number of channels at the bottleneck."

            # Initialize transformer bottleneck
            self.transformer_bottleneck = TransformerBottleneck(
                d_model=transformer_d_model,
                nhead=transformer_nhead,
                num_layers=transformer_num_layers,
                dim_feedforward=transformer_dim_feedforward,
                dropout=transformer_dropout
            )
        else:
            self.transformer_bottleneck = None

    def forward(self, x):
        # Input x shape: [batch_size, in_channels, spatial_size]

        #--------------------
        # Lift the input
        #--------------------
        x = self.lift(x)    # Transform input to higher-dimensional feature space
        skip = []           # List to store skip connections

        #--------------------
        # Encoder Path
        #--------------------
        for i in range(self.N_layers):
            # Apply ResNet at current resolution and store for skip connection
            y = self.res_nets[i](x)
            skip.append(y)

            # Downsample with encoder block
            x = self.encoder[i](x)

        #--------------------
        # Bottleneck
        #--------------------
        x = self.res_net_neck(x)    # Apply bottleneck ResNet

        # Apply transformer bottleneck if available
        if self.transformer_bottleneck is not None:
            x = self.transformer_bottleneck(x)

        #--------------------
        # Decoder Path
        #--------------------
        for i in range(self.N_layers):
            # Adjust skip connection features to match decoder dimensions
            if i == 0:
                # For the first decoder layer, just apply ED expansion
                x = self.ED_expansion[self.N_layers - i](x)
            else:
                # For subsequent layers, concatenate adjusted skip connections
                x = torch.cat(
                    (x, self.ED_expansion[self.N_layers - i](skip[-i])),
                    dim=1  # Concatenate along the channel dimension
                )
            # Upsample with decoder block
            x = self.decoder[i](x)

        #--------------------
        # Final Projection
        #--------------------
        # Concatenate the first skip connection after adjustment
        x = torch.cat(
            (x, self.ED_expansion[0](skip[0])),
            dim=1
        )
        x = self.project(x)   # Transform features back to output dimension

        return x               # Output shape: [batch_size, out_channels, spatial_size]