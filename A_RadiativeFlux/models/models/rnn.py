"""
MLP class for prediction over Icon grid data

Author: s.mohebi22[at]gmail.com
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .base_methods import BaseRadiationModel


class FastParallelLayers(nn.Module):
    """
    Fast implementation of parallel MLPs using 1D convolutions.
    
    This is more efficient than using separate MLPs for each height level.
    """
    def __init__(self, input_size, num_players, hidden_sizes, activation):
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
            self.activation = nn.SiLU()  # Default to SiLU/Swish

    def forward(self, x):
        x = x.reshape([-1, self.num_players*self.input_size, 1])
        for cnv in self.players:
            x = self.activation(cnv(x))
        return x.reshape([-1, self.num_players, self.output_size])


class FastRnnIg(BaseRadiationModel):
    """
    Fast RNN model for radiative transfer modeling with 1D convolutions.
    
    This model processes atmospheric column data using efficient 1D convolutions
    and bidirectional LSTMs, optimized for performance.
    """
    def __init__(
            self,
            height_in,
            channel_out,
            channel_3d,
            channel_2d,
            lstm_units,
            mlp_units,
            dropout,
            lstm_droprate,
            smoothing_kernel,
            beta,
            beta_height,
            beta_height_sw,
            beta_height_lw,
            device,
            *args, 
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        
        # Store configuration parameters
        self.height_in = height_in
        self.channel_out = channel_out
        self.smoothing_kernel = smoothing_kernel
        self.beta = beta
        self.beta_height = beta_height
        self.beta_height_sw = beta_height_sw
        self.beta_height_lw = beta_height_lw
        
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
        
        
    
    def _smooth(self, y, kernel_size):
        """
        Apply smoothing to the output using a moving average filter.
        
        Args:
            y: Output tensor [B, H, C]
            kernel_size: Size of the smoothing kernel
            
        Returns:
            Smoothed tensor with same shape as input
        """
        if kernel_size is None or kernel_size <= 1:
            return y
            
        # Create a simple moving average kernel
        kernel = torch.ones(1, 1, kernel_size, device=y.device) / kernel_size
        
        # Pad the input to maintain the same size
        padding = (kernel_size - 1) // 2
        
        # Process each channel separately
        smoothed_outputs = []
        for i in range(y.shape[-1]):
            channel = y[..., i]
            # [B, H] -> [B, 1, H]
            channel = channel.unsqueeze(1)
            # Apply padding
            channel_padded = F.pad(channel, (padding, padding), mode='replicate')
            # Apply convolution and squeeze back
            channel_smoothed = F.conv1d(channel_padded, kernel).squeeze(1)
            smoothed_outputs.append(channel_smoothed.unsqueeze(-1))
            
        # Concatenate all channels back together
        return torch.cat(smoothed_outputs, dim=-1)
      
      
    
    def exponential_decay(self, y):
        """
        Apply exponential decay to the output based on height.
        
        Args:
            y: Output tensor [B, H, C]
            
        Returns:
            Decayed tensor with same shape as input
        """
        if self.beta is None:
            return y
            
        # Create exponential decay factor
        batch_size, height, channels = y.shape
        decay = torch.exp(-torch.arange(height, device=y.device) * self.beta)
        
        # Apply different decay for different radiation channels if specified
        if self.beta_height_lw is not None and self.beta_height_sw is not None:
            decay_lw = torch.exp(-torch.arange(height, device=y.device) * self.beta_height_lw)
            decay_sw = torch.exp(-torch.arange(height, device=y.device) * self.beta_height_sw)
            
            # Apply to longwave (first two channels)
            y_decayed = y.clone()
            y_decayed[:, :, 0:2] = y[:, :, 0:2] * decay_lw.view(1, -1, 1)
            
            # Apply to shortwave (last two channels)
            y_decayed[:, :, 2:4] = y[:, :, 2:4] * decay_sw.view(1, -1, 1)
            
            return y_decayed
        
        # Apply uniform decay to all channels
        return y * decay.view(1, -1, 1)
      
      

    def forward(self, x3d_norm, x2d_norm, x2d_orig):
        """
        Forward pass with pre-normalized inputs.
        
        Args:
            x3d_norm: Normalized 3D input data [B, H, C3d]
            x2d_norm: Normalized 2D input data [B, C2d]
            x2d_orig: Original 2D input data (for scaling output) [B, C2d]
            
        Returns:
            Predicted radiative fluxes [B, H, C_out]
        """
        # Append a row of ones to match the expected height dimension
        one_vec = torch.ones((x3d_norm.shape[0], 1, x3d_norm.shape[-1]), device=x3d_norm.device)
        x3d_norm = torch.cat([x3d_norm, one_vec], dim=1)   # [B, H+1, C3d]
        
        # Repeat 2D features across height dimension
        x2d_repeated = x2d_norm.unsqueeze(1).repeat(1, x3d_norm.shape[1], 1)  # [B, H+1, C2d]
        
        # Concatenate 3D and 2D features
        x = torch.cat([x2d_repeated, x3d_norm], dim=-1)  # [B, H+1, C3d+C2d]
        
        # Process through first layer
        x = self.p_layer1(x)  # [B, H+1, mlp_units[-1]]
        
        # Process through LSTM layers
        for layer in self.bi_lstms:
            x, _ = layer(x)  # [B, H+1, lstm_units[-1]*2]
        
        # Process through output layer
        y = self.p_layer2(x)  # [B, H+1, channel_out]
        
        # Apply smoothing if requested
        if self.smoothing_kernel is not None:
            y = self._smooth(y, self.smoothing_kernel)
            
        # Apply decay if requested
        if self.beta is not None:
            y = self.exponential_decay(y)
        
        # Scale the output
        y = self._scale_output(y, x2d_orig)
            
        return y.squeeze()
