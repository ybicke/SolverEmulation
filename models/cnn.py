"""
Unet class for prediction over Icon grid data
Parts of the U-Net model 
Adapted from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
Author: s.mohebi22[at]gmail.com
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F

from .base_models import *


class DoubleConv(nn.Module):
  """(convolution => [BN] => ReLU) * 2"""

  def __init__(self, in_channels, out_channels, mid_channels=None):
    super().__init__()
    if not mid_channels:
        mid_channels = out_channels
    self.double_conv = nn.Sequential(
      nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm1d(mid_channels),
      nn.ReLU(inplace=True),
      nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm1d(out_channels),
      nn.ReLU(inplace=True)
    )

  def forward(self, x):
    return self.double_conv(x)


class Down(nn.Module):
  """Downscaling with maxpool then double conv"""
  def __init__(self, in_channels, out_channels, kernel_size=2):
    super().__init__()
    self.maxpool_conv = nn.Sequential(
      nn.MaxPool1d(kernel_size),
      DoubleConv(in_channels, out_channels)
    )

  def forward(self, x):
    return self.maxpool_conv(x)


class Up(nn.Module):
  """Upscaling then double conv"""
  def __init__(self, in_channels, out_channels):
    super().__init__()

    self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
    self.conv = DoubleConv(in_channels, out_channels)

  def forward(self, x1, x2):
    x1 = self.up(x1)
    diff = x2.size()[2] - x1.size()[2]
    # if you have padding issues, see
    # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
    # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
    x1 = F.pad(x1, [diff//2, diff-diff//2])
    x = torch.cat([x2, x1], dim=1)
    return self.conv(x)


class OutConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(OutConv, self).__init__()
    self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

  def forward(self, x):
    return self.conv(x)


class CnnIg(BaseIconModel):
  def __init__(self, x3d_mean, x3d_std, x2d_mean, x2d_std, args):
    super(CnnIg, self).__init__(x3d_mean, x3d_std, x2d_mean, x2d_std)

    height_out = args.height_out
    channel_out = args.channel_out
    channel_in = args.channel_2d + args.channel_3d
    height_in = args.height_in
    cnn_units = args.cnn_units
    kernel_sizes = args.cnn_maxpool_kernel_sizes
    self.scale_output = args.scale_output
    self.smoothing_kernel = args.smoothing_kernel
    self.beta = args.beta
    self.beta_height = args.beta_height
    self.beta_height_sw = args.beta_height_sw
    self.beta_height_lw = args.beta_height_lw

    self.inc = DoubleConv(channel_in, cnn_units[0])
    self.downs = nn.ModuleList()
    for i in range(0, len(cnn_units)-1):      
      self.downs.append(Down(cnn_units[i], cnn_units[i+1], kernel_sizes[i]))

    self.ups = nn.ModuleList()
    for i in range(len(cnn_units)-1, 0, -1):
      self.ups.append(Up(cnn_units[i], cnn_units[i-1]))
    self.outc = OutConv(cnn_units[0], channel_out)
    self.last_height = OutConv(height_in, height_out)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x3d, x2d):
    x2d_org = x2d.clone()
    
    x3d = self.normalizer3d(x3d)   # (b, 70, 6)
    x2d = self.normalizer2d(x2d)  # (b, 6)
        
    x2d = x2d.unsqueeze(1).repeat(1, x3d.shape[1], 1)   # (b, 70, 6)
    x = torch.cat([x2d, x3d], dim=-1)  # (b, 70, 12)

    xx = list()
    xx.append(self.inc(x.permute(0, 2, 1)))
    for down in self.downs:
      xx.append(down(xx[-1]))
    
    y = xx.pop()
    for up in self.ups:
      y = up(y, xx.pop())
    
    y = self.outc(y).permute(0, 2, 1)
    y = self.last_height(y)
    
    if self.scale_output:
      y = self.sigmoid(y)
      y = self._scale_output(y, x2d_org)
    if self.smoothing_kernel is not None:
      y = self._smooth(y, self.smoothing_kernel)
    if self.beta is not None:
      y = self.exponential_decay(y)
      
    return y


class CnnIg2(CnnIg):
  def __init__(self, x3d_mean, x3d_std, x2d_mean, x2d_std, args):
    super(CnnIg2, self).__init__(x3d_mean, x3d_std, x2d_mean, x2d_std, args)
    self.last_height = OutConv(args.height_in, 1)

  def forward(self, x3d, x2d):
    x2d_org = x2d.clone()

    x3d = self.normalizer3d(x3d)   # (b, 70, 6)
    x2d = self.normalizer2d(x2d)  # (b, 6)

    x2d = x2d.unsqueeze(1).repeat(1, x3d.shape[1], 1)   # (b, 70, 6)
    x = torch.cat([x2d, x3d], dim=-1)  # (b, 70, 12)

    xx = list()
    xx.append(self.inc(x.permute(0, 2, 1)))
    for down in self.downs:
      xx.append(down(xx[-1]))
  
    y = xx.pop()
    for up in self.ups:
      y = up(y, xx.pop())
  
    y = self.outc(y).permute(0, 2, 1)
    last = self.last_height(y)
    y = torch.concat([y, last], dim=1)
    if self.scale_output:
      y = self.sigmoid(y)
      y = self._scale_output(y, x2d_org)
    if self.smoothing_kernel is not None:
      y = self._smooth(y, self.smoothing_kernel)
    if self.beta is not None:
      y = self.exponential_decay(y)
    return y
    

class ClippedCnnIg2(CnnIg2):
  def __init__(self, x3d_mean, x3d_std, x2d_mean, x2d_std, args):
    super(ClippedCnnIg2, self).__init__(x3d_mean, x3d_std, x2d_mean, x2d_std, args)
    self.xclipp = args.xclipp

  def forward(self, x3d, x2d):
    x3d = torch.clamp(x3d, min=self.xclipp)
    x2d = torch.clamp(x2d, min=self.xclipp)

    x2d_org = x2d.clone()

    x3d = self.normalizer3d(x3d)   # (b, 70, 6)
    x2d = self.normalizer2d(x2d)  # (b, 6)

    x2d = x2d.unsqueeze(1).repeat(1, x3d.shape[1], 1)   # (b, 70, 6)
    x = torch.cat([x2d, x3d], dim=-1)  # (b, 70, 12)

    xx = list()
    xx.append(self.inc(x.permute(0, 2, 1)))
    for down in self.downs:
      xx.append(down(xx[-1]))
  
    y = xx.pop()
    for up in self.ups:
      y = up(y, xx.pop())
  
    y = self.outc(y).permute(0, 2, 1)
    last = self.last_height(y)
    y = torch.concat([y, last], dim=1)
    if self.scale_output:
      y = self.sigmoid(y)
      y = self._scale_output(y, x2d_org)
    if self.smoothing_kernel is not None:
      y = self._smooth(y, self.smoothing_kernel)
    if self.beta is not None:
      y = self.exponential_decay(y)
    return y