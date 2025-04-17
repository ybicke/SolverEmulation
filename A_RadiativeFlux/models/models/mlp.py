import torch 
import torch.nn as nn
import torch.nn.functional as F
from .base_models import *

class MlpIg(BaseIconModel):
  def __init__(self, x3d_mean, x3d_std, x2d_mean, x2d_std, args):
    super(MlpIg, self).__init__(x3d_mean, x3d_std, x2d_mean, x2d_std)
    height_in = args.height_in
    channel_3d = args.channel_3d
    channel_2d = args.channel_2d
    mlp_units = args.mlp_units
    head_units = args.mlp_head_units
    self.smoothing_kernel = args.smoothing_kernel
    self.beta = args.beta
    self.beta_height = args.beta_height
    self.beta_height_sw = args.beta_height_sw
    self.beta_height_lw = args.beta_height_lw

    self.height_out = args.height_out
    self.channel_out = args.channel_out
    self.scale_output = args.scale_output
    self.normalize_output = args.normalize_output
    self.y_mean = args.y_mean
    self.y_std = args.y_std

    self.flatten = nn.Flatten()

    self.x2d_layer = Mlp(
      input_size=channel_2d, 
      hidden_sizes=mlp_units, 
    )

    self.x3d_layer = ParallelLayers(
      input_size=channel_3d, 
      num_players=height_in, 
      hidden_sizes=mlp_units,
    )
    
    self.mlp_head = Mlp(
      input_size=mlp_units[-1] * (height_in + 1), 
      hidden_sizes=head_units, 
    )

    self.final = nn.Linear(head_units[-1], self.height_out * self.channel_out)
    self.sigmoid = nn.Sigmoid()
    self.relu = nn.ReLU()

  def forward(self, x3d, x2d):
    x2d_org = x2d.clone()
    x2d = self.normalizer2d(x2d)  # (b, 6)
    x3d = self.normalizer3d(x3d)  # (b, 70, 6)
    x2d = self.x2d_layer(x2d)  # (b, d)
    x3d = self.x3d_layer(x3d)  # (b, 70, d)
    x3d = self.flatten(x3d)  # (b, 70*d)
    x = torch.cat([x3d, x2d], dim=-1)  # (b, 71*d)
    x = self.mlp_head(x)  # (b, d2)
    y = self.final(x)  # (b, 284)
    y = y.reshape(-1, self.height_out, self.channel_out)  # (b, 71, 4)
    if self.scale_output:
      y = self.sigmoid(y)
      y = self._scale_output(y, x2d_org)
    # else:
      # y   = self.relu(y)
    # if self.normalize_output:
      # y = y * self.y_std + self.y_mean
    if self.smoothing_kernel is not None:
      y = self._smooth(y, self.smoothing_kernel)
    if self.beta is not None:
      y = self.exponential_decay(y)
    return y


class FastMlpIg(MlpIg):
  def __init__(self, x3d_mean, x3d_std, x2d_mean, x2d_std, args):
    super(FastMlpIg, self).__init__(x3d_mean, x3d_std, x2d_mean, x2d_std, args)
    height_in = args.height_in
    channel_3d = args.channel_3d
    mlp_units = args.mlp_units

    self.x3d_layer = FastParallelLayers(
      input_size=channel_3d, 
      num_players=height_in, 
      hidden_sizes=mlp_units,
    )
  

class MlpIgMaxPool(MlpIg):
  def __init__(self, x3d_mean, x3d_std, x2d_mean, x2d_std, args):
    super(MlpIgMaxPool, self).__init__(x3d_mean, x3d_std, x2d_mean, x2d_std, args)

    mlp_units = args.mlp_units
    height_in = args.height_in
    head_units = args.mlp_head_units
    kernel_size = args.mlp_maxpool_kernel_size
    activation = 'sigmoid' if args.scale_output else 'relu'

    self.maxpool = nn.MaxPool1d(kernel_size)

    self.mlp_head = Mlp(
      input_size=int(mlp_units[-1] * (height_in/kernel_size + 1)),
      hidden_sizes=head_units,
      activation=activation
    )

  def forward(self, x3d, x2d):
    x2d_org = x2d.clone()
    x2d = self.normalizer2d(x2d)  # (b, 6)
    x3d = self.normalizer3d(x3d)  # (b, 70, 6)
    x2d = self.x2d_layer(x2d)  # (b, d)
    x3d = self.x3d_layer(x3d)  # (b, 70, d)
    x3d = torch.permute(self.maxpool(torch.permute(x3d, [0, 2, 1])), [0, 2, 1]) # (b, 70/p, d)
    x3d = self.flatten(x3d)  # (b, 70*d)
    x = torch.cat([x3d, x2d], dim=-1)  # (b, 71*d)
    x = self.mlp_head(x)  # (b, d2)
    y = self.final(x)  # (b, 284)
    y = y.reshape(-1, self.height_out, self.channel_out)  # (b, 71, 4)
    if self.scale_output:
      y = self._scale_output(y, x2d_org)
    if self.smoothing_kernel is not None:
      y = self._smooth(y, self.smoothing_kernel)
    return y


class MlpSharedWeights(BaseIconModel):
  def __init__(self, x2d_mean, x2d_std, x3d_mean, x3d_std, args):
    super(MlpSharedWeights, self).__init__(x2d_mean, x2d_std, x3d_mean, x3d_std)
    height_in = args.height_in
    channel_3d = args.channel_3d
    channel_2d = args.channel_2d
    mlp_units = args.mlp_units
    head_units = args.mlp_head_units
    activation = 'sigmoid' if args.scale_output else 'relu'

    self.height_out = args.height_out
    self.channel_out = args.channel_out
    self.scale_output = args.scale_output       
    
    self.flatten = nn.Flatten()

    self.x2d_layer = Mlp(
      input_size=channel_2d, 
      hidden_sizes=mlp_units, 
      activation=activation
    )

    self.x3d_layer = Mlp(
      input_size=channel_3d, 
      hidden_sizes=mlp_units,
      activation=activation
    )
    
    self.mlp_head = Mlp(
      input_size=mlp_units[-1] * (height_in + 1), 
      hidden_sizes=head_units, 
      activation=activation
    )

    self.final = nn.Linear(head_units[-1], self.height_out * self.channel_out)

  def forward(self, x3d, x2d):
    x2d_org = x2d.clone()
    x2d = self.normalizer2d(x2d)  # (b, 6)
    x3d = self.normalizer3d(x3d)  # (b, 70, 6)
    x2d = self.x2d_layer(x2d)  # (b, d)
    x3d = self.x3d_layer(x3d)  # (b, 70, d)
    x3d = self.flatten(x3d)  # (b, d*70)
    x = torch.cat([x3d, x2d], dim=-1)  #   (b, 71*d)
    x = self.mlp_head(x)  # (b, d2)
    y = self.final(x)  # (b, 284)
    y = y.reshape(-1, self.height_out, self.channel_out)  # (b, 71, 4)
    if self.scale_output:
      y = self._scale_output(y, x2d_org)
    return y


class MlpBroadcasted(BaseIconModel):
  def __init__(self, x2d_mean, x2d_std, x3d_mean, x3d_std, args):
    super(MlpSharedWeights, self).__init__(x2d_mean, x2d_std, x3d_mean, x3d_std)
    height_in = args.height_in
    channel_3d = args.channel_3d
    channel_2d = args.channel_2d
    mlp_units = args.mlp_units
    head_units = args.mlp_head_units
    activation = 'sigmoid' if args.scale_output else 'relu'

    self.height_out = args.height_out
    self.channel_out = args.channel_out
    self.scale_output = args.scale_output       
    
    self.flatten = nn.Flatten()

    self.x2d_layer = Mlp(
      input_size=channel_2d, 
      hidden_sizes=mlp_units, 
      activation=activation
    )

    self.x3d_layer = Mlp(
      input_size=channel_3d, 
      hidden_sizes=mlp_units,
      activation=activation
    )
    
    self.mlp_head = Mlp(
      input_size=mlp_units[-1] * (height_in + 1), 
      hidden_sizes=head_units, 
      activation=activation
    )

    self.final = nn.Linear(head_units[-1], self.height_out * self.channel_out)

  def forward(self, x3d, x2d):
    x2d_org = x2d.clone()
    x2d = self.normalizer2d(x2d)  # (b, 6)
    x3d = self.normalizer3d(x3d)  # (b, 70, 6)
    x2d = self.x2d_layer(x2d)  # (b, d)
    x3d = self.x3d_layer(x3d)  # (b, 70, d)
    x3d = self.flatten(x3d)  # (b, d*70)
    x = torch.cat([x3d, x2d], dim=-1)  # (b, 71*d)
    x = self.mlp_head(x)  # (b, d2)
    y = self.final(x)  # (b, 284)
    y = y.reshape(-1, self.height_out, self.channel_out)  # (b, 71, 4)
    if self.scale_output:
      y = self._scale_output(y, x2d_org)
    return y


class EnsambledMlpIg(FastMlpIg):
  def __init__(self, x3d_mean, x3d_std, x2d_mean, x2d_std, args):
    super(EnsambledMlpIg, self).__init__(x3d_mean, x3d_std, x2d_mean, x2d_std, args)
    activation = 'sigmoid' if args.scale_output else 'relu'
    head_units = args.mlp_head_units
    n_ensambels = args.n_ensambels

    self.final = nn.Sequential(
      RepeatAlongDim1(n_ensambels),
      FastParallelLayers(
        input_size=head_units[-1], 
        num_players=n_ensambels, 
        hidden_sizes=[self.height_out * self.channel_out],
        activation=activation
        ),
        AverageSecondDimension()
      )


class ClippedMlpIg(FastMlpIg):
  def __init__(self, x3d_mean, x3d_std, x2d_mean, x2d_std, args):
    super(FastMlpIg, self).__init__(x3d_mean, x3d_std, x2d_mean, x2d_std, args)
    self.xclipp = args.xclipp

  def forward(self, x3d, x2d):
    x3d = torch.clamp(x3d, min=self.xclipp)
    x2d = torch.clamp(x2d, min=self.xclipp)
    x2d_org = x2d.clone()
    x2d = self.normalizer2d(x2d)  # (b, 6)
    x3d = self.normalizer3d(x3d)  # (b, 70, 6)
    x2d = self.x2d_layer(x2d)  # (b, d)
    x3d = self.x3d_layer(x3d)  # (b, 70, d)
    x3d = self.flatten(x3d)  # (b, 70*d)
    x = torch.cat([x3d, x2d], dim=-1)  # (b, 71*d)
    x = self.mlp_head(x)  # (b, d2)
    y = self.final(x)  # (b, 284)
    y = y.reshape(-1, self.height_out, self.channel_out)  # (b, 71, 4)
    if self.scale_output:
      y = self.sigmoid(y)
      y = self._scale_output(y, x2d_org)
    if self.smoothing_kernel is not None:
      y = self._smooth(y, self.smoothing_kernel)
    if self.beta is not None:
      y = self.exponential_decay(y)
    return y


class EnsambleOfFastMlpIg(MlpIg):
  def __init__(self, x2d_mean, x2d_std, x3d_mean, x3d_std, args):
    super(EnsambleOfFastMlpIg, self).__init__(x2d_mean, x2d_std, x3d_mean, x3d_std, args)
    self.ensambels = nn.ModuleList([
      FastMlpIg(x2d_mean, x2d_std, x3d_mean, x3d_std, args) for _ in range(args.n_ensambels)
    ])

  def forward(self, x3d, x2d):
    outputs = [model(x3d, x2d) for model in self.ensambels]
    outputs = torch.stack(outputs, dim=0)
    return torch.mean(outputs, dim=0)


class FastMlpIgSubFeats(MlpIg):
  def __init__(self, x3d_mean, x3d_std, x2d_mean, x2d_std, args):
    super(FastMlpIgSubFeats, self).__init__(x3d_mean, x3d_std, x2d_mean, x2d_std, args)
    height_in = args.height_in
    channel_3d = args.channel_3d
    mlp_units = args.mlp_units

    self.x3d_layer = FastParallelLayers(
      input_size=channel_3d-2, 
      num_players=height_in, 
      hidden_sizes=mlp_units,
    )

  def forward(self, x3d, x2d):
    x2d_org = x2d.clone()
    x2d = self.normalizer2d(x2d)  # (b, 6)
    x3d = self.normalizer3d(x3d)  # (b, 70, 6)
    x3d = x3d[:, :, [0, 1, 2, 5]]
    x2d = self.x2d_layer(x2d)  # (b, d)
    x3d = self.x3d_layer(x3d)  # (b, 70, d)
    x3d = self.flatten(x3d)  # (b, 70*d)
    x = torch.cat([x3d, x2d], dim=-1)  # (b, 71*d)
    x = self.mlp_head(x)  # (b, d2)
    y = self.final(x)  # (b, 284)
    y = y.reshape(-1, self.height_out, self.channel_out)  # (b, 71, 4)
    if self.scale_output:
      y = self.sigmoid(y)
      y = self._scale_output(y, x2d_org)
    if self.smoothing_kernel is not None:
      y = self._smooth(y, self.smoothing_kernel)
    if self.beta is not None:
      y = self.exponential_decay(y)
    return y
