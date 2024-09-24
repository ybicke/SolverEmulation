


import torch
import torch.nn as nn
import torch.nn.functional as F

#---------------------
# Activation Function:
#---------------------

class CNO_LReLu(nn.Module):
    def __init__(self, in_size, out_size):
        super(CNO_LReLu, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = F.interpolate(x.unsqueeze(2), size=(1, 2 * self.in_size), mode="bicubic", align_corners=False, antialias=True)
        x = self.act(x)
        x = F.interpolate(x, size=(1, self.out_size), mode="bicubic", align_corners=False, antialias=True)
        return x[:, :, 0]

#--------------------
# CNO Block:
#--------------------

class CNOBlock(nn.Module):
    def __init__(self, in_channels, out_channels, in_size, out_size, use_bn=True):
        super(CNOBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = in_size
        self.out_size = out_size

        # We apply Conv -> BN (optional) -> Activation
        # Up/Downsampling happens inside Activation

        self.convolution = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            padding=1
        )

        if use_bn:
            self.batch_norm = nn.BatchNorm1d(self.out_channels)
        else:
            self.batch_norm = nn.Identity()

        self.act = CNO_LReLu(in_size=self.in_size, out_size=self.out_size)

    def forward(self, x):
        x = self.convolution(x)
        x = self.batch_norm(x)
        return self.act(x)

#--------------------
# Lift/Project Block:
#--------------------

class LiftProjectBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size, latent_dim=64):
        super(LiftProjectBlock, self).__init__()

        self.inter_CNOBlock = CNOBlock(
            in_channels=in_channels,
            out_channels=latent_dim,
            in_size=size,
            out_size=size,
            use_bn=False
        )

        self.convolution = nn.Conv1d(
            in_channels=latent_dim,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):
        x = self.inter_CNOBlock(x)
        x = self.convolution(x)
        return x

#--------------------
# Residual Block:
#--------------------

class ResidualBlock(nn.Module):
    def __init__(self, channels, size, use_bn=True):
        super(ResidualBlock, self).__init__()

        self.channels = channels
        self.size = size

        # We apply Conv -> BN (optional) -> Activation -> Conv -> BN (optional) -> Skip Connection
        # Up/Downsampling happens inside Activation

        self.convolution1 = nn.Conv1d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=3,
            padding=1
        )
        self.convolution2 = nn.Conv1d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=3,
            padding=1
        )

        if use_bn:
            self.batch_norm1 = nn.BatchNorm1d(self.channels)
            self.batch_norm2 = nn.BatchNorm1d(self.channels)
        else:
            self.batch_norm1 = nn.Identity()
            self.batch_norm2 = nn.Identity()

        self.act = CNO_LReLu(in_size=self.size, out_size=self.size)

    def forward(self, x):
        residual = x
        x = self.convolution1(x)
        x = self.batch_norm1(x)
        x = self.act(x)
        x = self.convolution2(x)
        x = self.batch_norm2(x)
        x += residual
        return x

#--------------------
# ResNet:
#--------------------

class ResNet(nn.Module):
    def __init__(self, channels, size, num_blocks=4, use_bn=True):
        super(ResNet, self).__init__()
        self.num_blocks = num_blocks
        self.res_nets = nn.ModuleList(
            [ResidualBlock(channels=channels, size=size, use_bn=use_bn) for _ in range(num_blocks)]
        )

    def forward(self, x):
        for i in range(self.num_blocks):
            x = self.res_nets[i](x)
        return x

#--------------------
# Transformer Bottleneck:
#--------------------

class TransformerBottleneck(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerBottleneck, self).__init__()
        self.positional_encoding = nn.Parameter(torch.zeros(1, d_model, 1))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x shape: (batch_size, channels, sequence_length)
        x = x.permute(0, 2, 1)  # Reshape to (batch_size, sequence_length, channels)
        x += self.positional_encoding[:, :, :x.size(1)]
        x = self.transformer_encoder(x)
        x = x.permute(0, 2, 1)  # Reshape back to (batch_size, channels, sequence_length)
        return x

#--------------------
# Modified CNO1d with Transformer Bottleneck:
#--------------------

class CNO1d(nn.Module):
    def __init__(self,
                 in_dim,                  # Number of input channels.
                 out_dim,                 # Number of output channels.
                 size,                    # Input and Output spatial size.
                 N_layers,                # Number of (D) or (U) blocks in the network.
                 N_res=4,                 # Number of (R) blocks per level (except the neck).
                 N_res_neck=4,            # Number of (R) blocks in the neck.
                 channel_multiplier=16,   # Channel multiplier.
                 use_bn=True,             # Use BatchNorm.
                 transformer_d_model=None,      # Transformer model dimension.
                 transformer_nhead=None,         # Number of attention heads.
                 transformer_num_layers=None,    # Number of transformer layers.
                 transformer_dim_feedforward=2048,  # Feedforward network dimension.
                 transformer_dropout=0.1):       # Dropout rate.
        super(CNO1d, self).__init__()

        self.N_layers = int(N_layers)
        self.lift_dim = channel_multiplier // 2
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.channel_multiplier = channel_multiplier

        # Num of channels/features - encoder and decoder
        self.encoder_features = [self.lift_dim]
        for i in range(self.N_layers):
            self.encoder_features.append(2 ** i * self.channel_multiplier)

        self.decoder_features_in = self.encoder_features[1:]
        self.decoder_features_in.reverse()
        self.decoder_features_out = self.encoder_features[:-1]
        self.decoder_features_out.reverse()

        for i in range(1, self.N_layers):
            self.decoder_features_in[i] = 2 * self.decoder_features_in[i]  # Adjust for concatenation

        # Spatial sizes of channels - encoder and decoder
        self.encoder_sizes = []
        self.decoder_sizes = []
        for i in range(self.N_layers + 1):
            self.encoder_sizes.append(size // 2 ** i)
            self.decoder_sizes.append(size // 2 ** (self.N_layers - i))

        # Define Lift and Project blocks
        self.lift = LiftProjectBlock(
            in_channels=self.in_dim,
            out_channels=self.encoder_features[0],
            size=size
        )

        self.project = LiftProjectBlock(
            in_channels=self.encoder_features[0] + self.decoder_features_out[-1],
            out_channels=self.out_dim,
            size=size
        )

        # Define Encoder, ED Linker, and Decoder networks
        self.encoder = nn.ModuleList([
            CNOBlock(
                in_channels=self.encoder_features[i],
                out_channels=self.encoder_features[i + 1],
                in_size=self.encoder_sizes[i],
                out_size=self.encoder_sizes[i + 1],
                use_bn=use_bn
            )
            for i in range(self.N_layers)
        ])

        self.ED_expansion = nn.ModuleList([
            CNOBlock(
                in_channels=self.encoder_features[i],
                out_channels=self.encoder_features[i],
                in_size=self.encoder_sizes[i],
                out_size=self.decoder_sizes[self.N_layers - i],
                use_bn=use_bn
            )
            for i in range(self.N_layers + 1)
        ])

        self.decoder = nn.ModuleList([
            CNOBlock(
                in_channels=self.decoder_features_in[i],
                out_channels=self.decoder_features_out[i],
                in_size=self.decoder_sizes[i],
                out_size=self.decoder_sizes[i + 1],
                use_bn=use_bn
            )
            for i in range(self.N_layers)
        ])

        # Define ResNet Blocks
        self.res_nets = nn.ModuleList([
            ResNet(
                channels=self.encoder_features[l],
                size=self.encoder_sizes[l],
                num_blocks=N_res,
                use_bn=use_bn
            )
            for l in range(self.N_layers)
        ])

        self.res_net_neck = ResNet(
            channels=self.encoder_features[self.N_layers],
            size=self.encoder_sizes[self.N_layers],
            num_blocks=N_res_neck,
            use_bn=use_bn
        )

        # Transformer Bottleneck
        if (
            transformer_d_model is not None and
            transformer_nhead is not None and
            transformer_num_layers is not None
        ):
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
        x = self.lift(x)  # Execute Lift
        skip = []

        # Execute Encoder
        for i in range(self.N_layers):
            # Apply ResNet & save the result
            y = self.res_nets[i](x)
            skip.append(y)

            # Apply (D) block
            x = self.encoder[i](x)

        # Apply the deepest ResNet (bottleneck)
        x = self.res_net_neck(x)

        # Apply the transformer bottleneck if available
        if self.transformer_bottleneck is not None:
            x = self.transformer_bottleneck(x)

        # Execute Decoder
        for i in range(self.N_layers):
            # Apply (I) block (ED_expansion) & concatenate if needed
            if i == 0:
                x = self.ED_expansion[self.N_layers - i](x)  # Bottleneck: no concatenation
            else:
                x = torch.cat(
                    (x, self.ED_expansion[self.N_layers - i](skip[-i])),
                    dim=1
                )

            # Apply (U) block
            x = self.decoder[i](x)

        # Concatenate & Execute Projection
        x = torch.cat((x, self.ED_expansion[0](skip[0])), dim=1)
        x = self.project(x)

        return x