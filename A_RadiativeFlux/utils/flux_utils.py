#!/usr/bin/env python3
"""
Utility functions for flux modeling including heating rate calculations
and smoothness loss.
"""

import torch
from torchmetrics import Metric


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


class SmoothHeatingRateSmoothnessLoss(torch.nn.Module):
    """Heating rate smoothness loss with smooth transition (RNN-friendly)"""
    def __init__(self, weight=0.1, full_start_level=30, transition_end_level=50, mode='1d', transition_type='linear'):
        super(SmoothHeatingRateSmoothnessLoss, self).__init__()
        self.weight = weight
        self.full_start_level = full_start_level  # Level where full weight starts (atmospheric top)
        self.transition_end_level = transition_end_level  # Level where weight becomes 0 (towards surface)
        self.mode = mode
        self.transition_type = transition_type
        
    def forward(self, outputs, x3d, x2d):
        hr_pred = calculate_heating_rates(outputs, x3d, x2d, mode=self.mode)
        hr_diff_pred = hr_pred[:, 1:, :] - hr_pred[:, :-1, :]
        
        n_levels = hr_diff_pred.shape[1]  # Should be 70
        
        # Create weight function based on atmospheric levels
        # Level 0 = atmospheric top, Level 70 = surface
        weights = torch.zeros(n_levels, device=hr_diff_pred.device)
        
        # Full weight from level 0 to full_start_level (e.g., 0-29)
        if self.full_start_level > 0:
            weights[:self.full_start_level] = 1.0
        
        # Smooth transition from full_start_level to transition_end_level (e.g., 30-49)
        if self.transition_end_level > self.full_start_level:
            transition_length = self.transition_end_level - self.full_start_level
            
            if self.transition_type == 'linear':
                # Linear transition from 1 to 0
                weights[self.full_start_level:self.transition_end_level] = torch.linspace(
                    1, 0, transition_length, device=hr_diff_pred.device
                )
            elif self.transition_type == 'sigmoid':
                # Sigmoid transition from 1 to 0
                x = torch.linspace(3, -3, transition_length, device=hr_diff_pred.device)
                weights[self.full_start_level:self.transition_end_level] = torch.sigmoid(x)
            elif self.transition_type == 'quadratic':
                # Quadratic transition (fast start, slow finish)
                x = torch.linspace(1, 0, transition_length, device=hr_diff_pred.device)
                weights[self.full_start_level:self.transition_end_level] = x**2
        
        # Zero weight from transition_end_level to surface (e.g., 50-70)
        # (already zero by default)
        
        # Apply weights: [B, levels, features] * [levels, 1]
        weights = weights.unsqueeze(0).unsqueeze(-1)  # [1, levels, 1]
        hr_diff_weighted = hr_diff_pred * weights
        smoothness_loss = torch.mean(hr_diff_weighted**2)
            
        return self.weight * smoothness_loss 


class EnergyConservationLossV2(Metric):
    """
    Physics-informed loss that combines MSE with energy conservation constraint.
    
    This loss function enforces energy conservation by ensuring that the heating rates
    calculated from predicted fluxes match those from true fluxes, promoting physical
    consistency in the predictions.
    
    Args:
        alpha (float): Weighting factor between MSE loss (alpha) and energy conservation loss (1-alpha)
        mode (str): '1d' or '3d' mode for proper data handling
        dist_sync_on_step (bool): Whether to synchronize metric state across processes
    """
    def __init__(self, alpha=0.5, mode='1d', dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.alpha = alpha
        self.mode = mode
        self.add_state("mse_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("ec_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor, x3d: torch.Tensor, x2d: torch.Tensor):
        """
        Update the metric state with a new batch of predictions.
        
        Args:
            y_true: True flux values
            y_pred: Predicted flux values  
            x3d: 3D input data (for pressure and other atmospheric variables)
            x2d: 2D input data (for surface pressure and other surface variables)
        """
        mse_loss = torch.mean(torch.square(y_true - y_pred))
        
        # Calculate heating rates using the existing function
        hr_true = calculate_heating_rates(y_true, x3d, x2d, mode=self.mode)
        hr_pred = calculate_heating_rates(y_pred, x3d, x2d, mode=self.mode)
        
        ec_loss = torch.mean(torch.square(hr_true - hr_pred))
        total_loss = self.alpha * mse_loss + (1 - self.alpha) * ec_loss
        
        self.mse_loss += mse_loss
        self.ec_loss += ec_loss
        self.total_loss += total_loss
        self.total += y_true.shape[0]

    def compute(self):
        """Compute the final metric value."""
        return self.total_loss / self.total
    
    def get_components(self):
        """Get individual loss components for logging."""
        return {
            'mse_component': self.mse_loss / self.total,
            'ec_component': self.ec_loss / self.total,
            'total_loss': self.total_loss / self.total
        } 