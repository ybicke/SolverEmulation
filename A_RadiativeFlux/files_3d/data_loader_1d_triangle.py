import math
import shutil
import random
import torch
import h5py
from os.path import join, basename, exists
from torch.utils.data import IterableDataset
from files_3d.data_utils import get_triangle_indices

class IconTriangleColumnDataset(IterableDataset):
    """
    Loads columns from a specific triangle area but treats each column independently.
    This combines features of both IconColumnIterableDataset and IconIterableDataset_3D:
    - Selects columns only from a specific triangle area
    - Returns each column independently (not the whole triangle as one sample)
    - Supports subsampling within the triangle area
    
    This allows testing whether the model performance difference is due to:
    - Data distribution (from the same area vs global sampling)
    - Implementation differences
    """
    def __init__(
        self,
        triangle_id,
        division_factor,
        filenames,
        shuffle=False,
        subsample=None,  # Added subsample parameter
        dtype='float32',
        cache_dir=None,
        total_cols=81920,

    ):
        super().__init__()
        self.filenames = filenames
        self.cache_dir = cache_dir
        self.shuffle = shuffle
        self.triangle_id = triangle_id
        self.subsample = subsample  # Store subsample parameter
        
        # Get the indices of all columns in the specified triangle area
        self.triangle_indices = get_triangle_indices(
            triangle_id, 
            division_factor=division_factor,
            total_cols=total_cols

        )
        
        # Convert dtype string to torch dtype
        self.dtype = getattr(torch, dtype)
    
    def read_file(self, filename):
        """Read and preprocess an H5 file, returning columns from the triangle area"""
        local_file = filename
        if self.cache_dir:
            local_file = join(self.cache_dir, basename(filename))
            if not exists(local_file):
                shutil.copy2(filename, local_file)
        
        for _ in range(10):  # Try up to 10 times if there are file access issues
            try:
                with h5py.File(local_file, 'r') as h:
                    # Load only data for columns in our triangle area
                    x3d_full = torch.tensor(h['x3d'][:], dtype=self.dtype)
                    x2d_full = torch.tensor(h['x2d'][:], dtype=self.dtype)
                    y_full   = torch.tensor(h['y'][:],   dtype=self.dtype)
                    
                # Extract triangle columns
                x3d = x3d_full[self.triangle_indices]
                x2d = x2d_full[self.triangle_indices]
                y   = y_full[self.triangle_indices]
                
                # Apply subsampling if specified
                if self.subsample is not None and self.subsample < 1.0:
                    num_samples = int(self.subsample * len(self.triangle_indices))
                    if num_samples > 0:
                        indices = torch.randperm(len(self.triangle_indices))[:num_samples]
                        x3d = x3d[indices]
                        x2d = x2d[indices]
                        y = y[indices]
                
                # Return all (or subsampled) columns from the triangle
                return x3d, x2d, y
                
            except Exception as exc:
                print(f"Error reading {local_file}: {exc}")
                if self.cache_dir:
                    shutil.copy2(filename, local_file)
                continue
            break
            
        # If we reach here, all attempts failed
        raise RuntimeError(f"Failed to read file {filename} after multiple attempts")
    
    def __iter__(self):
        """Iterator that yields individual columns from triangle area"""
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
            
            # Return each column as a separate sample
            # This is different from data_loader_3d which returns the entire triangle as one sample
            for col_idx in range(len(x3d)):
                yield x3d[col_idx], x2d[col_idx], y[col_idx]

    def __len__(self):
        # Approximate length calculation with subsample consideration
        approx_columns_per_file = len(self.triangle_indices)
        if self.subsample is not None and self.subsample < 1.0:
            approx_columns_per_file = int(approx_columns_per_file * self.subsample)
        return len(self.filenames) * approx_columns_per_file 