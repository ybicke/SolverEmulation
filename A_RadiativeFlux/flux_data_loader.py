import math
import shutil
import random
import torch
import h5py
from os.path import join, basename, exists
from torch.utils.data import IterableDataset
from utils.data_utils import get_triangle_indices


class UnifiedFluxDataset(IterableDataset):
    """
    Unified dataset loader for radiative flux training that supports:
    - Both 1D (column-wise) and 3D (triangle-wise) modes
    - Both triangle and full dataset selection
    - Consistent data preprocessing for both modes
    
    Modes:
    - 1D: Each atmospheric column is a separate sample (for column-wise models)
    - 3D: Entire selected region (triangle or full) is one sample (for graph-based models)
    
    Dataset types:
    - triangle: Select a specific triangular region of the globe
    - full: Use all columns from the entire globe
    
    This unifies the functionality of:
    - IconColumnIterableDataset (1D, full globe)
    - IconTriangleColumnDataset (1D, triangle area)
    - IconIterableDataset_3D (3D, triangle area)
    """
    def __init__(
        self,
        filenames,
        mode='1d',  # '1d' for column-wise, '3d' for triangle-wise
        dataset_type='triangle',  # 'triangle' or 'full'
        triangle_id=0,
        division_factor=1,
        shuffle=False,
        subsample=None,
        dtype='float32',
        cache_dir=None,
        total_cols=81920,
        subsample_seed=42,  # Add deterministic seed for subsampling
    ):
        super().__init__()
        self.filenames = filenames
        self.cache_dir = cache_dir
        self.shuffle = shuffle
        self.mode = mode
        self.dataset_type = dataset_type
        self.triangle_id = triangle_id
        self.division_factor = division_factor
        self.subsample = subsample
        self.subsample_seed = subsample_seed  # Store seed for deterministic subsampling
        
        # Convert dtype string to torch dtype
        self.dtype = getattr(torch, dtype)
        
        # Set up triangle indices based on dataset type
        if dataset_type == 'triangle':
            self.triangle_indices = get_triangle_indices(
                triangle_id, 
                division_factor=division_factor,
                total_cols=total_cols
            )
            self.is_triangle_mode = True
            print(f"Triangle {triangle_id} with division_factor {division_factor}: "
                  f"{len(self.triangle_indices)} columns (indices {self.triangle_indices[0]} to {self.triangle_indices[-1]})")
        elif dataset_type == 'full':
            self.triangle_indices = None
            self.is_triangle_mode = False
            print(f"Full dataset mode: loading all {total_cols} columns")
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}. Must be 'triangle' or 'full'")
        
        # Print mode information
        if mode == '1d':
            print(f"Mode: 1D (each column is a separate sample)")
        else:  # mode == '3d'
            print(f"Mode: 3D (entire region processed as single graph)")

    def read_file(self, filename):
        """Read and preprocess H5 files, returning data from selected area"""
        local_file = filename
        
        # Optimize caching - only copy if not already cached
        if self.cache_dir:
            local_file = join(self.cache_dir, basename(filename))
            if not exists(local_file):
                shutil.copy2(filename, local_file)
        
        for _ in range(10):  # Try up to 10 times if there are file access issues
            try:
                with h5py.File(local_file, 'r') as h:
                    if self.is_triangle_mode:
                        # Load only triangle columns to save memory and time
                        x3d = torch.tensor(h['x3d'][self.triangle_indices, :, :], dtype=self.dtype)
                        x2d = torch.tensor(h['x2d'][self.triangle_indices], dtype=self.dtype)
                        y = torch.tensor(h['y'][self.triangle_indices, :, :], dtype=self.dtype)
                    else:
                        # Load all columns
                        x3d = torch.tensor(h['x3d'][:, :, :], dtype=self.dtype)
                        x2d = torch.tensor(h['x2d'][:], dtype=self.dtype)
                        y = torch.tensor(h['y'][:, :, :], dtype=self.dtype)
                
                # Apply subsampling if specified - DETERMINISTIC VERSION
                if self.subsample is not None and self.subsample < 1.0:
                    num_samples = int(self.subsample * x3d.shape[0])
                    if num_samples > 0:
                        # Simple deterministic subsampling - same pattern for all files
                        rng = torch.Generator()
                        rng.manual_seed(self.subsample_seed)
                        indices = torch.randperm(x3d.shape[0], generator=rng)[:num_samples]
                        
                        x3d = x3d[indices]
                        x2d = x2d[indices]
                        y = y[indices]
                
                return x3d, x2d, y
                
            except Exception as exc:
                print(f"Error reading {local_file}: {exc}")
                # Only re-copy on error, not every retry
                if self.cache_dir:
                    shutil.copy2(filename, local_file)
                continue
            break
            
        # If we reach here, all attempts failed
        raise RuntimeError(f"Failed to read file {filename} after multiple attempts")
    
    def __iter__(self):
        """Iterator that yields data based on mode (1D or 3D)"""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self.filenames)
        else:
            per_worker = int(math.ceil(len(self.filenames) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.filenames))
        
        file_indices = list(range(iter_start, iter_end))
       
        if self.shuffle:
            random.shuffle(file_indices)
        
        # Process each file
        for idx in file_indices:
            filename = self.filenames[idx]
            x3d, x2d, y = self.read_file(filename)
            
            if self.mode == '1d':
                # 1D mode: Return each column as a separate sample
                for col_idx in range(len(x3d)):
                    yield x3d[col_idx], x2d[col_idx], y[col_idx]
            elif self.mode == '3d':
                # 3D mode: Return entire triangle/dataset as one sample
                yield x3d, x2d, y
            else:
                raise ValueError(f"Unknown mode: {self.mode}. Must be '1d' or '3d'")


    def __len__(self):
        """Approximate length calculation based on mode"""
        if self.is_triangle_mode:
            approx_columns_per_file = len(self.triangle_indices)
        else:
            approx_columns_per_file = 81920  # Total number of columns
            
        if self.subsample is not None and self.subsample < 1.0:
            approx_columns_per_file = int(approx_columns_per_file * self.subsample)
        
        if self.mode == '1d':
            # 1D mode: Each column is a separate sample
            return len(self.filenames) * approx_columns_per_file
        elif self.mode == '3d':
            # 3D mode: Each file is one sample
            return len(self.filenames)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


# Backward compatibility aliases
IconColumnIterableDataset = UnifiedFluxDataset
IconTriangleColumnDataset = UnifiedFluxDataset
IconIterableDataset_3D = UnifiedFluxDataset 