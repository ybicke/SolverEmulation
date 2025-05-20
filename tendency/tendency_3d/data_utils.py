"""
Utilities for data processing and normalization.
"""

import torch

class DataNormalizer:
    """
    Utility class for normalizing data for radiation models.
    This class handles normalization of 3D and 2D input data using
    pre-calculated mean and variance statistics.
    """
    
    def __init__(self, mean2d, var2d, mean3d, var3d, device='cuda'):
        """
        Initialize the DataNormalizer with mean and variance statistics.
        
        Args:
            mean2d: Mean values for 2D data
            var2d: Variance values for 2D data
            mean3d: Mean values for 3D data
            var3d: Variance values for 3D data
            device: Device to perform calculations on (default: 'cuda')
        """
        self.mean2d = mean2d
        self.std2d = torch.sqrt(var2d)
        self.mean3d = mean3d
        self.std3d = torch.sqrt(var3d)
        self.device = device
    
    def normalize(self, x3d, x2d):
        """
        Normalize 3D and 2D data using mean and standard deviation values.
        
        Args:
            x3d: 3D input data (batch_size, height, channels)
            x2d: 2D input data (batch_size, channels)
            
        Returns:
            x3d_normalized: Normalized 3D data
            x2d_normalized: Normalized 2D data
            x2d_original: Original 2D data
        """
        # Store original 2D data for scaling output
        x2d_original = x2d.clone()
        
        # For 4D input [batch, columns, height, channels]
        if len(x3d.shape) == 4:
            batch_size, num_columns, height, channels = x3d.shape
            # Normalize while preserving 4D structure
            x3d_normalized = (x3d - self.mean3d.view(1, 1, 1, -1)) / self.std3d.view(1, 1, 1, -1)
        elif len(x3d.shape) == 3:
            # Handle 3D case as before
            x3d_normalized = (x3d - self.mean3d.view(1, 1, -1)) / self.std3d.view(1, 1, -1)
        
        # Normalize 2D data - similar logic for dimension handling
        if len(x2d.shape) == 2:  # [batch, channels]
            x2d_normalized = (x2d - self.mean2d.view(1, -1)) / self.std2d.view(1, -1)
        elif len(x2d.shape) == 3:  # [batch, columns, channels]
            # Preserve 3D structure instead of reshaping
            x2d_normalized = (x2d - self.mean2d.view(1, 1, -1)) / self.std2d.view(1, 1, -1)
        
        return x3d_normalized, x2d_normalized, x2d_original



def get_triangle_indices(triangle_id, division_factor, total_cols):
    """
    Get indices of triangle cells for a specific triangle ID with optional division.
    
    Args:
        triangle_id: ID of the triangle to select (0-19 for large triangles)
        total_cols: Total number of columns in the grid
        division_factor: Factor by which to divide the triangle 
                        (1 for full triangle with 4096 columns, 
                         4 for 1/4 triangle with 1024 columns,
                         16 for 1/16 triangle with 256 columns)
                         
    Returns:
        torch.Tensor: Indices for the selected triangle area
    """
    # First get the large triangle size
    large_triangle_size = total_cols // 20
    
    # Calculate block size for subdivided triangles
    block_size = large_triangle_size // division_factor
    
    # Which of the 20 large triangles are we in?
    large_triangle_id = triangle_id // division_factor
    
    # Which subdivision within that large triangle?
    sub_block_index = triangle_id % division_factor
    
    # Calculate start index
    start_idx = (large_triangle_id * large_triangle_size) + (sub_block_index * block_size)
    
    end_idx = min(start_idx + block_size, total_cols)
    return torch.arange(start_idx, end_idx, dtype=torch.long)

def interpolate_w_to_full_levels(w):
    """
    Linear interpolation of vertical velocity from half levels to full levels.
    Works with both 3D data [B,N,L,1] and 2D data [B,L,1].
    
    Args:
        w: Vertical velocity at half levels
        
    Returns:
        Interpolated vertical velocity at full levels
    """
    if len(w.shape) == 4:  # 3D data [B,N,L,1]
        batch_size, num_columns, num_half_levels, _ = w.shape
        num_full_levels = num_half_levels - 1
        w_full = torch.zeros((batch_size, num_columns, num_full_levels, 1), device=w.device)
        w_full[:, :, :, :] = 0.5 * (w[:, :, :-1, :] + w[:, :, 1:, :])
    else:  # 2D data [B,L,1]
        batch_size, num_half_levels, _ = w.shape
        num_full_levels = num_half_levels - 1
        w_full = torch.zeros((batch_size, num_full_levels, 1), device=w.device)
        w_full[:, :, :] = 0.5 * (w[:, :-1, :] + w[:, 1:, :])
        
    return w_full 