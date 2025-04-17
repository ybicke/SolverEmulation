import torch
from torch import nn
from einops import rearrange, repeat

from .base_methods import BaseRadiationModel

class Normalization(nn.Module):
    """ Normalize the input based on mean and std """    
    def __init__(self, std, mean):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, x):
        assert x.size(dim=-1) == self.std.size(dim=0), f'Dimension mismatch'
        return (x - self.mean)/self.std

class FastParallelLayers(nn.Module):
    def __init__(self, input_size, num_players, hidden_sizes, activation='relu'):
        super().__init__()
        f = input_size
        l = num_players
        self.num_players = num_players
        self.input_size = input_size
        self.output_size = hidden_sizes[-1]

        self.players = nn.ModuleList()
        for unit in hidden_sizes:
            self.players.append(
                nn.Conv1d(in_channels=l*f, out_channels=l*unit, kernel_size=1, groups=l)
            )
            f = unit

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError

    def forward(self, x):
        x = x.reshape([-1, self.num_players*self.input_size, 1])
        for cnv in self.players:
            x = self.activation(cnv(x))
        return x.reshape([-1, self.num_players, self.output_size])

class FastRnnIg(BaseRadiationModel):
    def __init__(
            self,
            dropout=0.0, 
            height_in=70,
            channel_out=4,
            channel_3d=6,
            channel_2d=6,
            lstm_units=[256, 512],
            mlp_units=[128, 256],
            lstm_droprate=0.0,
            device=None,
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)
        
        # Store configuration parameters
        self.height_in = height_in
        self.channel_out = channel_out
        
        # Define network layers
        self.p_layer1 = FastParallelLayers(
            input_size=channel_3d+channel_2d, 
            num_players=height_in+1, 
            hidden_sizes=mlp_units,
            activation='sigmoid'
        )

        # Calculate LSTM input sizes
        lstm_input_sizes = mlp_units[-1:] + [2*e for e in lstm_units]
        
        # LSTM layers
        self.bi_lstms = nn.ModuleList()
        for i in range(len(lstm_units)):
            self.bi_lstms.append(
                nn.LSTM(
                    input_size=lstm_input_sizes[i],
                    hidden_size=lstm_units[i],
                    batch_first=True,
                    dropout=lstm_droprate,
                    bidirectional=True
                )
            )
        
        # Output layer
        self.p_layer2 = FastParallelLayers(
            input_size=lstm_input_sizes[-1],
            num_players=height_in+1,
            hidden_sizes=[channel_out],
            activation='sigmoid'
        )
    

    def forward(self, x3d_norm, x2d_norm, x2d_orig):
        # Append a row of ones to match the expected height dimension
        one_vec = torch.ones((x3d_norm.shape[0], 1, x3d_norm.shape[-1]), device=x3d_norm.device)
        x3d_norm = torch.cat([x3d_norm, one_vec], dim=1)   # (b, 71, 6)
        
        # Repeat 2D features across height dimension
        x2d_repeated = x2d_norm.unsqueeze(1).repeat(1, x3d_norm.shape[1], 1)
        
        # Concatenate 3D and 2D features
        x = torch.cat([x2d_repeated, x3d_norm], dim=-1)
        
        # Process through first layer
        x = self.p_layer1(x)
        
        # Process through LSTM layers
        for layer in self.bi_lstms:
            x, _ = layer(x)
        
        # Process through output layer
        y = self.p_layer2(x)
        
        # Scale the output
        y = self._scale_output(y, x2d_orig)
            
        return y