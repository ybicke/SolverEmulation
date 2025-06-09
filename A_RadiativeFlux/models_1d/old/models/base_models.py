"""
NN models for prediction over Icon grid columns

Author: s.mohebi22[at]gmail.com
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F


# Helper classes 
class Normalization(nn.Module):
  """ Normalize the input based on mean and std """    
  def __init__(self, std, mean, axis=None):
    super().__init__()
    self.std = std
    self.mean = mean
    self.axis = axis

  def forward(self, x):
    return (x - self.mean) / (self.std + 1e-8)


class Mlp(nn.Module):
  """ Multi-layer perceptron """
  def __init__(self, input_size, hidden_sizes, activation='relu'):
    super().__init__()

    self.input_layer = nn.Linear(input_size, hidden_sizes[0])

    self.hidden_layers = nn.ModuleList()
    for i in range(len(hidden_sizes) - 1):
      self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))

    if activation == 'relu':
      self.activation = nn.ReLU()
    elif activation == 'sigmoid':
      self.activation = nn.Sigmoid()
    else:
      raise NotImplementedError

  def forward(self, x):
    x = self.activation(self.input_layer(x))
    for hidden_layer in self.hidden_layers:
      x = self.activation(hidden_layer(x))
    return x
    

class ParallelLayers(nn.Module):
  def __init__(self, input_size, num_players, hidden_sizes, activation='relu'):
    super().__init__()
    self.num_players = num_players  
    self.pmlp = nn.ModuleList()
    self.last_dim = hidden_sizes[-1]

    for _ in range(num_players): # num parallel layers
      self.pmlp.append(Mlp(input_size, hidden_sizes, activation=activation))
    
  def forward(self, x):
    x2 = torch.zeros(x.shape[0], self.num_players, self.last_dim, device=x.device)
    for i, pmlp in enumerate(self.pmlp):
      x2[:, i, :] = pmlp(x[:, i, :])
    return x2


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


class BaseIconModel(nn.Module):
  def __init__(self, x3d_mean, x3d_std, x2d_mean, x2d_std):
    super(BaseIconModel, self).__init__()
    self.lwflx_idx = [0, 1]
    self.swflx_idx = [2, 3]
    self.cosmu0_idx = 1
    self.tsfctrad_idx = 5
    self.normalizer2d = Normalization(std=x2d_std, mean=x2d_mean)
    self.normalizer3d = Normalization(std=x3d_std, mean=x3d_mean)
    
  def _unscale_swflx(self, swflx, cosmu0):
    "Returns the scaled swflx to the original scale"
    return torch.where(
      cosmu0 >= torch.tensor(1e-4, dtype=torch.float32),
      swflx * (cosmu0 * 1400), 0
    )

  def _unscale_lwflx(self, lwflx, tsfctrad):
    "Returns the scaled lwflx to the original scale"
    stefan_boltzmann_const = torch.tensor(5.670374419e-08, dtype=torch.float32)
    return torch.where(
      tsfctrad >= torch.tensor(1e-4, dtype=torch.float32),
      lwflx * torch.pow(tsfctrad, 4) * stefan_boltzmann_const,
      lwflx
    )
    
  def _scale_output(self, y_pred, x2d_org):
    y_pred_scaled = []

    for i in range(y_pred.shape[-1]):
      f_pred = y_pred[..., i:i+1]

      if i in self.swflx_idx:
        cosmu0 = x2d_org[..., self.cosmu0_idx]
        cosmu0 = torch.tile(
          cosmu0[..., None, None], (1, f_pred.shape[-2], 1)
        )
        y_pred_scaled.append(self._unscale_swflx(f_pred, cosmu0))
      elif i in self.lwflx_idx:
        tsfctrad = x2d_org[..., self.tsfctrad_idx]
        tsfctrad = torch.tile(
          tsfctrad[..., None, None], (1, f_pred.shape[-2], 1)
        )
        y_pred_scaled.append(self._unscale_lwflx(f_pred, tsfctrad))
      else:
        y_pred_scaled.append(f_pred)

    y_pred = torch.cat(y_pred_scaled, dim=-1)
    y_pred[:, 0, 1] = 0
    return y_pred

  def _smooth(self, y_pred, kernel):
    y_pred = torch.einsum('abc,bd->adc', y_pred, kernel).contiguous()
    return y_pred
  
  def exponential_decay(self, y):
    
    beta_lw = self.beta[:self.beta_height_lw, :2]
    beta_sw = self.beta[:self.beta_height_sw, 2:]

    cbeta_lw = torch.cumprod(beta_lw.flip([0]), dim=0).flip(dims=(0,))
    cbeta_sw = torch.cumprod(beta_sw.flip([0]), dim=0).flip(dims=(0,))
    

    y_lw_bottom = y[:, self.beta_height_lw:, :2]
    y_sw_bottom = y[:, self.beta_height_sw:, 2:]

    y_lw_top = y_lw_bottom[:, 0:1, :] * cbeta_lw.unsqueeze(0)
    y_sw_top = y_sw_bottom[:, 0:1, :] * cbeta_sw.unsqueeze(0)
    
    y_lw = torch.cat([y_lw_top, y_lw_bottom], dim=1)
    y_sw = torch.cat([y_sw_top, y_sw_bottom], dim=1)
    
    y_new = torch.cat([y_lw, y_sw], dim=-1)
    return y_new


class RepeatAlongDim1(nn.Module):
  def __init__(self, b):
    super(RepeatAlongDim1, self).__init__()
    self.b = b

  def forward(self, x):
    x = x.unsqueeze(1)
    x = x.repeat(1, self.b, 1)
    return x


class AverageSecondDimension(nn.Module):
  def __init__(self):
    super(AverageSecondDimension, self).__init__()

  def forward(self, x):
    return x.mean(dim=1)
