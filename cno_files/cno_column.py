import torch
import torch.nn as nn
import torch.nn.functional as F

# Make sure the modified CNO1d class with the transformer bottleneck is defined in your code
# or imported correctly if it's in a separate file.

# If you have the Normalization class and other helper functions, include them as well.

class Normalization(nn.Module):
    """Normalize the input based on mean and std."""
    def __init__(self, std, mean, axis=None):
        super().__init__()
        self.std = std
        self.mean = mean
        self.axis = axis

    def forward(self, x):
        # Supports normalization on last dim
        assert x.size(-1) == self.std.size(0), \
            f'Dimension mismatch ({x.size(-1)} != {self.std.size(0)})'
        return (x - self.mean) / self.std

class CNONet(nn.Module):
    """
    CNONet class using the CNO1d model with a transformer bottleneck
    for predicting radiative flux.
    """

    def __init__(self,
                 in_dim,              # Number of input channels (e.g., 6)
                 out_dim,             # Number of output channels (e.g., 4)
                 size,                # Spatial size of input/output (e.g., 71)
                 N_layers=4,
                 N_res=4,
                 N_res_neck=4,
                 channel_multiplier=16,
                 use_bn=True,
                 transformer_d_model=None,
                 transformer_nhead=None,
                 transformer_num_layers=None,
                 transformer_dim_feedforward=2048,
                 transformer_dropout=0.1,
                 swflx_idx=[2, 3],
                 lwflx_idx=[0, 1],
                 cosmu0_idx=1,
                 tsfctrad_idx=5,
                 mean2d=None,
                 var2d=None,
                 mean3d=None,
                 var3d=None,
                 device=None,
                 *args,
                 **kwargs):
        super(CNONet, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.size = size
        self.device = device
        self.swflx_idx = swflx_idx
        self.lwflx_idx = lwflx_idx
        self.cosmu0_idx = cosmu0_idx
        self.tsfctrad_idx = tsfctrad_idx

        # Initialize normalizers for 2D and 3D inputs
        self.normalizer2d = Normalization(std=torch.sqrt(var2d), mean=mean2d)
        self.normalizer3d = Normalization(std=torch.sqrt(var3d), mean=mean3d)

        # Initialize the CNO1d model
        self.cno_model = CNO1d(
            in_dim=in_dim,
            out_dim=out_dim,
            size=size,
            N_layers=N_layers,
            N_res=N_res,
            N_res_neck=N_res_neck,
            channel_multiplier=channel_multiplier,
            use_bn=use_bn,
            transformer_d_model=transformer_d_model,
            transformer_nhead=transformer_nhead,
            transformer_num_layers=transformer_num_layers,
            transformer_dim_feedforward=transformer_dim_feedforward,
            transformer_dropout=transformer_dropout
        )

    # Radiation task specifics:

    def _unscale_swflx(self, swflx, cosmu0):
        return torch.where(
            cosmu0 >= 1e-4,
            swflx * (cosmu0 * 1400),
            torch.zeros_like(swflx)
        )

    def _unscale_lwflx(self, lwflx, tsfctrad):
        stefan_boltzmann_const = 5.670374419e-08
        return torch.where(
            tsfctrad >= 1e-4,
            lwflx * torch.pow(tsfctrad, 4) * stefan_boltzmann_const,
            lwflx
        )

    def _scale_output(self, y_pred, x2d):
        y_pred_scaled = []

        for i in range(y_pred.shape[1]):  # Assuming y_pred shape: (batch_size, channels_out, sequence_length)
            f_pred = y_pred[:, i:i+1, :]  # Shape: (batch_size, 1, sequence_length)
            if i in self.swflx_idx:
                cosmu0 = x2d[:, self.cosmu0_idx].unsqueeze(-1)  # Shape: (batch_size, 1)
                f_pred_scaled = self._unscale_swflx(f_pred, cosmu0)
            elif i in self.lwflx_idx:
                tsfctrad = x2d[:, self.tsfctrad_idx].unsqueeze(-1)  # Shape: (batch_size, 1)
                f_pred_scaled = self._unscale_lwflx(f_pred, tsfctrad)
            else:
                f_pred_scaled = f_pred
            y_pred_scaled.append(f_pred_scaled)
        y_pred_scaled = torch.cat(y_pred_scaled, dim=1)  # Shape: (batch_size, channels_out, sequence_length)
        return y_pred_scaled

    def forward(self, x3d, x2d):
        # x3d: (batch_size, sequence_length, channels_in_3d)
        # x2d: (batch_size, channels_in_2d)

        # Normalize inputs
        x3d = self.normalizer3d(x3d)
        x2d = self.normalizer2d(x2d)

        # Prepare input for CNO1d
        # Transpose x3d to shape (batch_size, channels_in_3d, sequence_length)
        x3d = x3d.permute(0, 2, 1)

        # Expand x2d to match the sequence_length and concatenate with x3d
        sequence_length = x3d.size(2)
        x2d_expanded = x2d.unsqueeze(2).expand(-1, -1, sequence_length)  # Shape: (batch_size, channels_in_2d, sequence_length)

        # Combine x3d and x2d_expanded along the channel dimension
        x_input = torch.cat((x3d, x2d_expanded), dim=1)  # Shape: (batch_size, in_dim, sequence_length)

        # Pass through the CNO1d model
        y_pred = self.cno_model(x_input)  # Output shape: (batch_size, out_dim, sequence_length)

        # Apply activation function (e.g., sigmoid)
        y_pred = torch.sigmoid(y_pred)

        # Scale output
        y_pred_scaled = self._scale_output(y_pred, x2d)

        # Permute output to match expected shape: (batch_size, sequence_length, out_dim)
        y_pred_scaled = y_pred_scaled.permute(0, 2, 1)

        # If necessary, squeeze the sequence_length dimension if it's 1
        if y_pred_scaled.size(1) == 1:
            y_pred_scaled = y_pred_scaled.squeeze(1)

        return y_pred_scaled