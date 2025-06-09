"""
Base model functionality for radiation models

This module provides common base classes and methods used across different model types
in the radiation emulation models.
"""

import torch
from torch import nn


class BaseRadiationModel(nn.Module):
    """Base class for all radiation models with common scaling functionality"""
    
    def __init__(
            self,
            swflx_idx=[2, 3],
            lwflx_idx=[0, 1], 
            cosmu0_idx=1, 
            tsfctrad_idx=5,
            *args, **kwargs
        ):
        super().__init__()
        
        # These indices are used for scaling calculations
        self.swflx_idx = swflx_idx
        self.lwflx_idx = lwflx_idx
        self.cosmu0_idx = cosmu0_idx
        self.tsfctrad_idx = tsfctrad_idx


    def _unscale_swflx(self, swflx, cosmu0):
        """Returns the scaled shortwave flux to the original scale"""
        return torch.where(
            cosmu0 >= torch.tensor(1e-4, dtype=torch.float32),
            swflx * (cosmu0 * 1400),
            0
        )

    def _unscale_lwflx(self, lwflx, tsfctrad):
        """Returns the scaled longwave flux to the original scale"""
        stefan_boltzmann_const = torch.tensor(5.670374419e-08, dtype=torch.float32)
        return torch.where(
            tsfctrad >= torch.tensor(1e-4, dtype=torch.float32),
            lwflx * torch.pow(tsfctrad, 4) * stefan_boltzmann_const,
            lwflx
        )
    
    def _scale_output(self, y_pred, x2d):
        """Scale the model output based on the input 2D features"""
        y_pred_scaled = []

        for i in range(y_pred.shape[-1]):
            f_pred = y_pred[..., i:i+1]
            
            if i in self.swflx_idx:
                cosmu0 = x2d[..., self.cosmu0_idx]
                
                cosmu0 = torch.tile(
                    cosmu0[..., None, None], 
                    (1, 1, f_pred.shape[-2], 1)
                )
                y_pred_scaled.append(self._unscale_swflx(f_pred, cosmu0))
                
            elif i in self.lwflx_idx:
                tsfctrad = x2d[..., self.tsfctrad_idx]
                tsfctrad = torch.tile(
                    tsfctrad[..., None, None], 
                    (1, 1, f_pred.shape[-2], 1)
                )
                y_pred_scaled.append(self._unscale_lwflx(f_pred, tsfctrad))
            else:
                y_pred_scaled.append(f_pred)

        y_pred = torch.cat(y_pred_scaled, dim=-1)
        return y_pred
    
