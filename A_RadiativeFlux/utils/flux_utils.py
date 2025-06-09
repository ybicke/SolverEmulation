#!/usr/bin/env python3
"""
Utility functions for flux modeling including heating rate calculations
and smoothness loss.
"""

import torch


def calculate_heating_rates(y, x3d, x2d, mode='1d'):
    """Calculate heating rates from flux predictions
    
    Args:
        y: Flux predictions (batch, [columns,] levels, features)
        x3d: 3D input data (batch, [columns,] levels, features)  
        x2d: 2D input data (batch, [columns,] features)
        mode: '1d' or '3d' to handle different data shapes
        
    Returns:
        heating_rate: Calculated heating rates
    """
    # assumed output order is: [lw_up, lw_dn, sw_up, sw_dn]
    g = 9.80665
    cd = 1005
    cv = 1855
    qv_ecrad_in_idx = 5
    pres_ecrad_in_idx = 2
    pres_sfc_ecrad_in_idx = 0

    # Handle different data shapes based on mode
    if mode == '1d':
        # 1D mode: (batch, levels, features)
        qv = x3d[..., qv_ecrad_in_idx]
        pres = torch.zeros([x3d.shape[0], 71], device=y.device)
        pres[..., 1:] = x3d[..., pres_ecrad_in_idx]
        pres[..., 0] = x2d[..., pres_sfc_ecrad_in_idx]
    else:
        # 3D mode: (batch, columns, levels, features)
        qv = x3d[..., qv_ecrad_in_idx]
        batch_size, num_columns = x3d.shape[0], x3d.shape[1]
        pres = torch.zeros([batch_size, num_columns, 71], device=y.device)
        pres[..., 1:] = x3d[..., pres_ecrad_in_idx]
        pres[..., 0] = x2d[..., pres_sfc_ecrad_in_idx]
    
    pres[...,1:-1] = torch.sqrt(pres[...,1:-1] * pres[...,2:])
    top_interp = pres[...,-1] / torch.sqrt(pres[...,-2]*pres[...,-1])
    pres[...,1:-1] = torch.sqrt(pres[...,0:69] * pres[...,1:-1])
    pres[...,-1] = top_interp
    pres = torch.unsqueeze(pres, dim=-1)
    qv = torch.unsqueeze(qv, dim=-1)
    
    heating_rate = ((-g / (cd * (1-qv) + cv * qv)) / \
    (pres[..., :-1, :] - pres[...,1:,:])) * \
    ((y[..., :-1, [0, 2]] - y[..., :-1, [1, 3]]) - \
    (y[..., 1:, [0, 2]] - y[..., 1:, [1, 3]])) * 24*60*60
    return heating_rate


class HeatingRateSmoothnessLoss(torch.nn.Module):
    """Heating rate smoothness regularization loss using L2"""
    def __init__(self, weight=0.1, top_levels=None, mode='1d'):
        super(HeatingRateSmoothnessLoss, self).__init__()
        self.weight = weight
        self.top_levels = top_levels
        self.mode = mode
        
    def forward(self, outputs, x3d, x2d):
        hr_pred = calculate_heating_rates(outputs, x3d, x2d, mode=self.mode)
        hr_diff_pred = hr_pred[:, 1:, :] - hr_pred[:, :-1, :]
        
        if self.top_levels is not None:
            n_levels = min(self.top_levels, hr_diff_pred.shape[1])
            hr_diff_pred = hr_diff_pred[:, -n_levels:, :]
            
        smoothness_loss = torch.mean(hr_diff_pred**2)
        return self.weight * smoothness_loss 