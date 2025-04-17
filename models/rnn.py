"""
MLP class for prediction over Icon grid data

Author: s.mohebi22[at]gmail.com
"""

import torch 
import torch.nn as nn

from .base_models import *


class RnnIg(BaseIconModel):
  def __init__(self, x3d_mean, x3d_std, x2d_mean, x2d_std, args):
    super(RnnIg, self).__init__(x3d_mean, x3d_std, x2d_mean, x2d_std)

    height_in = args.height_in
    channel_out = args.channel_out
    channel_3d = args.channel_3d
    channel_2d = args.channel_2d
    mlp_units = args.mlp_units
    lstm_units = args.lstm_units
    lstm_droprate = args.lstm_droprate
    self.scale_output = args.scale_output
    activation = 'sigmoid' if args.scale_output else 'relu'
    self.smoothing_kernel = args.smoothing_kernel
    self.beta = args.beta
    self.beta_height = args.beta_height
    self.beta_height_sw = args.beta_height_sw
    self.beta_height_lw = args.beta_height_lw
   
    self.p_layer1 = ParallelLayers(
      input_size=channel_3d+channel_2d, 
      num_players=height_in+1, 
      hidden_sizes=mlp_units,
      activation=activation
    )

    self.bi_lstms = nn.ModuleList()
    lstm_input_sizes = mlp_units[-1:] + [2*e for e in lstm_units]
    for i in range(len(lstm_units)):
      self.bi_lstms.append(
        nn.LSTM(
          input_size=lstm_input_sizes[i],
          hidden_size=lstm_units[i],
          # num_layers=height_in+1,
          batch_first=True,
          dropout=lstm_droprate,
          bidirectional=True
        )
    )
    
    self.p_layer2 = ParallelLayers(
      input_size=lstm_input_sizes[-1],
      num_players=height_in+1,
      hidden_sizes=[channel_out],
      activation='sigmoid'
    )

  def forward(self, x3d, x2d):
    x3d = self.normalizer3d(x3d)   # (b, 70, 6)

    one_vec = torch.ones((x3d.shape[0], 1, x3d.shape[-1]), device=x3d.device)
    x3d = torch.cat([x3d, one_vec], dim=-2)   # (b, 71, 6)
    
    x2d_org = x2d.clone()
    x2d = self.normalizer2d(x2d)  # (b, 6)
    
    x2d = x2d.unsqueeze(1).repeat(1, x3d.shape[1], 1)   # (b, 71, 6)

    x = torch.cat([x2d, x3d], dim=-1)        
    x = self.p_layer1(x)  # (b, 71, 256)
    for layer in self.bi_lstms:
      x, _ = layer(x)

    y = self.p_layer2(x)  # (b, 71, 4)
    if self.scale_output:
      y = self._scale_output(y, x2d_org)
    if self.smoothing_kernel is not None:
      y = self._smooth(y, self.smoothing_kernel)
    if self.beta is not None:
      y = self.exponential_decay(y)
    return y
  

class FastRnnIg(RnnIg):
  def __init__(self, x3d_mean, x3d_std, x2d_mean, x2d_std, args):
    super(FastRnnIg, self).__init__(x3d_mean, x3d_std, x2d_mean, x2d_std, args)

    height_in = args.height_in
    channel_out = args.channel_out
    channel_3d = args.channel_3d
    channel_2d = args.channel_2d
    mlp_units = args.mlp_units
    lstm_units = args.lstm_units
    self.scale_output = args.scale_output
    activation = 'sigmoid' if args.scale_output else 'relu'
    
    self.p_layer1 = FastParallelLayers(
      input_size=channel_3d+channel_2d, 
      num_players=height_in+1, 
      hidden_sizes=mlp_units,
      activation=activation
    )

    lstm_input_sizes = mlp_units[-1:] + [2*e for e in lstm_units]
    

    self.p_layer2 = FastParallelLayers(
      input_size=lstm_input_sizes[-1],
      num_players=height_in+1,
      hidden_sizes=[channel_out],
      activation='sigmoid'
    )

class NewRnnIg(BaseIconModel):
  def __init__(self, x3d_mean, x3d_std, x2d_mean, x2d_std, args):
    super(NewRnnIg, self).__init__(x3d_mean, x3d_std, x2d_mean, x2d_std)

    height_in = args.height_in
    channel_out = args.channel_out
    channel_3d = args.channel_3d
    channel_2d = args.channel_2d
    mlp_units = args.mlp_units
    lstm_units = args.lstm_units
    lstm_droprate = args.lstm_droprate
    self.scale_output = args.scale_output
    activation = 'sigmoid' if args.scale_output else 'relu'
    self.smoothing_kernel = args.smoothing_kernel
    self.beta = args.beta
    self.beta_height = args.beta_height
    self.beta_height_sw = args.beta_height_sw
    self.beta_height_lw = args.beta_height_lw
   
    self.mlp = Mlp(channel_2d, mlp_units, activation='relu')
    
    self.p_layer1 = FastParallelLayers(
      input_size=channel_3d, 
      num_players=height_in+1, 
      hidden_sizes=mlp_units,
      activation=activation
    )
    self.layer_norm = nn.LayerNorm(
      normalized_shape=(height_in+1, 2*mlp_units[-1]), eps=1e-6
    )
    self.bi_lstms = nn.ModuleList()
    lstm_input_sizes = [2*mlp_units[-1]] + [2*e for e in lstm_units]
    
    for i in range(len(lstm_units)):
      self.bi_lstms.append(
        nn.LSTM(
          input_size=lstm_input_sizes[i],
          hidden_size=lstm_units[i],
          # num_layers=height_in+1,
          batch_first=True,
          dropout=lstm_droprate,
          bidirectional=True
        )
    )
        
    self.p_layer2 = FastParallelLayers(
      input_size=lstm_input_sizes[-1],
      num_players=height_in+1,
      hidden_sizes=[channel_out],
      activation='sigmoid'
    )

  def forward(self, x3d, x2d):
    x3d = self.normalizer3d(x3d)   # (b, 70, 6)

    one_vec = torch.ones((x3d.shape[0], 1, x3d.shape[-1]), device=x3d.device)
    x3d = torch.cat([x3d, one_vec], dim=-2)   # (b, 71, 6)
    
    x2d_org = x2d.clone()
    x2d = self.normalizer2d(x2d)  # (b, 6)
     
    x2d = self.mlp(x2d) # (b, D)
    x3d = self.p_layer1(x3d)  # (b, 71, 256)
    
    x2d = x2d.unsqueeze(1).repeat(1, x3d.shape[1], 1)   # (b, 71, D)

    x = torch.cat([x2d, x3d], dim=-1)
    
    x = self.layer_norm(x)

    for i, layer in enumerate(self.bi_lstms):
      x, _ = layer(x)

    y = self.p_layer2(x)  # (b, 71, 4)

    if self.scale_output:
      y = self._scale_output(y, x2d_org)
    if self.smoothing_kernel is not None:
      y = self._smooth(y, self.smoothing_kernel)
    if self.beta is not None:
      y = self.exponential_decay(y)
    return y


class NewRnnIgSharedWeights(NewRnnIg):
  def __init__(self, x3d_mean, x3d_std, x2d_mean, x2d_std, args):
    super(NewRnnIgSharedWeights, self).__init__(x3d_mean, x3d_std, x2d_mean, x2d_std, args)
    activation = 'sigmoid' if args.scale_output else 'relu'
    channel_3d = args.channel_3d
    mlp_units = args.mlp_units
    channel_out = args.channel_out
    lstm_units = args.lstm_units

    self.p_layer1 = Mlp(channel_3d, mlp_units, activation=activation)
    lstm_input_sizes = [2*mlp_units[-1]] + [2*e for e in lstm_units]
        
    self.p_layer2 = Mlp(
      input_size=lstm_input_sizes[-1],
      hidden_sizes=[channel_out],
      activation=activation
    )   


class ClippedNewRnnIg(NewRnnIg):
  def __init__(self, x3d_mean, x3d_std, x2d_mean, x2d_std, args):
    super(ClippedNewRnnIg, self).__init__(x3d_mean, x3d_std, x2d_mean, x2d_std, args)
    self.xclipp = args.xclipp

  def forward(self, x3d, x2d):
    x3d = torch.clamp(x3d, min=self.xclipp)
    x2d = torch.clamp(x2d, min=self.xclipp)

    x3d = self.normalizer3d(x3d)   # (b, 70, 6)

    one_vec = torch.ones((x3d.shape[0], 1, x3d.shape[-1]), device=x3d.device)
    x3d = torch.cat([x3d, one_vec], dim=-2)   # (b, 71, 6)
    
    x2d_org = x2d.clone()
    x2d = self.normalizer2d(x2d)  # (b, 6)
     
    x2d = self.mlp(x2d) # (b, D)
    x3d = self.p_layer1(x3d)  # (b, 71, 256)
    
    x2d = x2d.unsqueeze(1).repeat(1, x3d.shape[1], 1)   # (b, 71, D)

    x = torch.cat([x2d, x3d], dim=-1)
    
    x = self.layer_norm(x)

    for i, layer in enumerate(self.bi_lstms):
      x, _ = layer(x)

    y = self.p_layer2(x)  # (b, 71, 4)

    if self.scale_output:
      y = self._scale_output(y, x2d_org)
    if self.smoothing_kernel is not None:
      y = self._smooth(y, self.smoothing_kernel)
    if self.beta is not None:
      y = self.exponential_decay(y)
    return y
