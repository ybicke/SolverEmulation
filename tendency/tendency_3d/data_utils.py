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
        
        # Normalize 3D data - apply normalization along the channel dimension
        # Using proper broadcasting by ensuring dimensions are properly aligned
        x3d_normalized = (x3d - self.mean3d.view(1, 1, 1, -1)) / self.std3d.view(1, 1, 1, -1)        
        # Normalize 2D data
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
    # First get the large triangle size (4096 columns)
    large_triangle_size = total_cols // 20
    
    # Calculate actual block size based on division factor
    block_size = large_triangle_size // division_factor
    
    # Calculate start index within the large triangle
    large_triangle_start = (triangle_id // 20) * large_triangle_size
    
    # Calculate sub-block index within the large triangle
    if division_factor > 1:
        # For a division factor of 4, there are 4 possible sub-blocks (0,1,2,3)
        sub_block_index = triangle_id % division_factor
        start_idx = large_triangle_start + (sub_block_index * block_size)
    else:
        start_idx = large_triangle_start
    
    end_idx = min(start_idx + block_size, total_cols)
    return torch.arange(start_idx, end_idx, dtype=torch.long) 