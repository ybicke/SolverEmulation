import math
import shutil
import random
import torch
import h5py
import numpy as np
from os.path import join, basename, exists
from torch.utils.data import IterableDataset
from ..data_utils import get_triangle_indices


class IconIterableDataset_3D(IterableDataset):
    """
    Loads the entire large triangle (no subsampling) from each H5 "time chunk."
    Each iteration yields one sample: (x3d, x2d, y) for the entire triangle at that time.
    """
    def __init__(
        self,
        filenames,
        triangle_id=1,
        shuffle=False,
        dtype='float32',
        cache_dir=None,
        total_cols=81920,
        division_factor=1
    ):
        super().__init__()
        self.filenames = filenames
        self.cache_dir = cache_dir
        self.shuffle = shuffle
        self.triangle_id = triangle_id
        self.triangle_indices = get_triangle_indices(
            triangle_id, 
            total_cols=total_cols,
            division_factor=division_factor
        )
        
        # Convert dtype string to torch dtype
        self.dtype = getattr(torch, dtype)
    
    def read_file(self, filename):
        local_file = filename
        if self.cache_dir:
            local_file = join(self.cache_dir, basename(filename))
            if not exists(local_file):
                shutil.copy2(filename, local_file)
        
        for _ in range(10):
            try:
                with h5py.File(local_file, 'r') as h:
                    x3d_full = torch.tensor(h['x3d'][:], dtype=self.dtype)
                    x2d_full = torch.tensor(h['x2d'][:], dtype=self.dtype)
                    y_full   = torch.tensor(h['y'][:],   dtype=self.dtype)
                    
                # Keep only the large triangle columns
                x3d = x3d_full[self.triangle_indices]
                x2d = x2d_full[self.triangle_indices]
                y   = y_full[self.triangle_indices]
                
            except Exception as exc:
                print(f"Error reading {local_file}: {exc}")
                if self.cache_dir:
                    shutil.copy2(filename, local_file)
                continue
            break
        
        # Return the entire triangle (all columns, all vertical levels).
        return x3d, x2d, y
    
    def __iter__(self):
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
        
        # Each file => 1 sample: the entire triangle
        for idx in file_indices:
            filename = self.filenames[idx]
            x3d, x2d, y = self.read_file(filename)
            yield x3d, x2d, y

    def __len__(self):
        # Some folks define len as number of files, 
        # or skip it if not strictly needed
        return len(self.filenames)


############################
# Usage / Example
############################
# if __name__ == "__main__":
#     # e.g. each H5 is one 3-hour snapshot with shape [81920, ...]
#     h5_files = [
#         "/path/to/time_step_0001.h5",
#         "/path/to/time_step_0002.h5",
#         # ...
#     ]
    
    # Pick triangle 0 => columns [0..4095] if total_cols=81920
    # dataset = IconTriangleTimeDataset(
    #     filenames=h5_files,
    #     triangle_id=0,
    #     shuffle=False,
    #     dtype='float32',
    #     cache_dir=None,
    #     total_cols=81920
    # )
    
    # loader = DataLoader(dataset, batch_size=1, num_workers=0)
    
    # for i, (x3d_batch, x2d_batch, y_batch) in enumerate(loader):
    #     # Here, x3d_batch has shape [1, num_cols_in_triangle, vertical_levels, ...]
    #     # or whatever shape your data is in
    #     # We keep batch_size=1 => each sample = entire large triangle at that time
    #     print(f"Time step {i}, x3d shape = {x3d_batch.shape}")
    #     # Now pass (x3d_batch, x2d_batch) to your GNN, which sees the entire triangle's adjacency.
    #     # ...
