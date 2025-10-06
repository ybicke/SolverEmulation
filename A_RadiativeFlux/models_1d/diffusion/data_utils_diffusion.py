"""
Utilities for data processing, normalization, and dataset loading.
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
from data_loader import IconColumnIterableDataset

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
        Handles both individual samples and batched data.
        
        Args:
            x3d: 3D input data (can be [height, channels] or [batch, height, channels])
            x2d: 2D input data (can be [channels] or [batch, channels])
            
        Returns:
            x3d_normalized: Normalized 3D data
            x2d_normalized: Normalized 2D data
            x2d_original: Original 2D data
        """
        # Store original 2D data for scaling output
        x2d_original = x2d.clone()
        
        # Ensure stats are on CPU if the input tensors are on CPU
        device = x3d.device
        mean3d = self.mean3d.to(device)
        std3d = self.std3d.to(device)
        mean2d = self.mean2d.to(device)
        std2d = self.std2d.to(device)
        
        # Check if inputs are individual samples or batches
        if x3d.dim() == 2:  # [height, channels] - individual sample
            x3d_normalized = (x3d - mean3d.view(1, -1)) / std3d.view(1, -1)
            x2d_normalized = (x2d - mean2d) / std2d
        else:  # [batch, height, channels] - batched data
            x3d_normalized = (x3d - mean3d.view(1, 1, -1)) / std3d.view(1, 1, -1)
            x2d_normalized = (x2d - mean2d.view(1, -1)) / std2d.view(1, -1)
        
        return x3d_normalized, x2d_normalized, x2d_original


class IconDiffusionDataset(IterableDataset):
    """
    Adapts the IconColumnIterableDataset to work with the Lightning EDM framework.
    Wraps data in the expected format for diffusion models.
    """
    def __init__(self, filenames, normalizer, shuffle=None, subsample=None, cache_dir=None):
        """
        Initialize the dataset adapter.
        
        Args:
            filenames: List of data files to use
            normalizer: DataNormalizer instance for normalizing data
            shuffle: Whether to shuffle the files
            subsample: Subsampling rate
            cache_dir: Directory for caching data
        """
        self.icon_dataset = IconColumnIterableDataset(
            filenames=filenames,
            shuffle=shuffle,
            subsample=subsample,
            cache_dir=cache_dir
        )
        self.normalizer = normalizer
        
    def __iter__(self):
        for x3d, x2d, y in self.icon_dataset:
            # Ensure tensors first
            if not isinstance(x3d, torch.Tensor):
                x3d = torch.tensor(x3d)
                x2d = torch.tensor(x2d)
                y = torch.tensor(y)
            
            # Normalize with batch dimension
            x3d_norm, x2d_norm, x2d_orig = self.normalizer.normalize(x3d, x2d)
            
            # Create the batch in the format required by EDM
            batch = {
                "sample": y,
                "cond": {
                    "x3d_norm": x3d_norm,
                    "x2d_norm": x2d_norm,
                    "x2d_orig": x2d_orig
                }
            }
            
            yield batch
    
    # Removing __len__ to avoid warnings and confusion with multi-worker dataloaders
    # When using IterableDataset with multiple workers, __len__ can be misleading 