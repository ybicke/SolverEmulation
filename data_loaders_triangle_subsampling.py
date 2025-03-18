import math
import shutil
import random
import torch
import h5py
from torch.utils.data import IterableDataset, DataLoader
from os.path import join, basename, exists
import torch


# TODO: Need to check if this is correctly subsampling into smaller triangle
def get_subtriangle_indices(
    triangle_id=0, 
    subtriangle_id=0, 
    subtriangle_count=4, 
    total_cols=81920
):
    """
    Returns a tensor of column indices corresponding to a 'subtriangle'
    within a larger naive 'triangle' block.

    Example:
        - total_cols=81920 => 20 blocks => each block has 4096 columns
        - triangle_id=0 => picks the block columns [0..4095]
        - subtriangle_count=4 => we divide 4096 => 4 sub-blocks of 1024 each
        - subtriangle_id=0..3 => picks one of those sub-blocks

    Returns:
        torch.LongTensor of column indices
    """
    # 1) Large triangle block
    block_size = total_cols // 20  # 4096 if total_cols=81920
    triangle_start = triangle_id * block_size
    triangle_end   = triangle_start + block_size  # typically start+4096

    # 2) Sub-splitting that block
    sub_size = block_size // subtriangle_count
    start_idx = triangle_start + subtriangle_id * sub_size
    end_idx   = start_idx + sub_size

    # Make sure we don't exceed the triangle_end
    end_idx = min(end_idx, triangle_end)

    return torch.arange(start_idx, end_idx, dtype=torch.long)




class IconSubTriangleIterableDataset(IterableDataset):
    """
    Loads a sub-block of columns from a chosen naive large triangle
    across multiple time-chunk H5 files.
    """

    def __init__(
        self,
        filenames,
        triangle_id=0,
        subtriangle_id=0,
        subtriangle_count=4,
        shuffle=False, 
        dtype='float32', 
        cache_dir=None,
        total_cols=81920
    ):
        super().__init__()
        self.filenames = filenames
        self.cache_dir = cache_dir
        self.shuffle = shuffle
        
        # Convert dtype string to torch dtype
        self.dtype = getattr(torch, dtype)

        # Precompute the sub-block indices
        self.col_indices = get_subtriangle_indices(
            triangle_id=triangle_id,
            subtriangle_id=subtriangle_id,
            subtriangle_count=subtriangle_count,
            total_cols=total_cols
        )

    def read_file(self, filename):
        """
        Reads one H5 file, selects only the sub-block columns.
        Returns the entire time-chunk of data for that sub-block.
        """
        local_file = filename
        if self.cache_dir:
            local_file = join(self.cache_dir, basename(filename))
            if not exists(local_file):
                shutil.copy2(filename, local_file)
        
        # Retry logic
        for _ in range(10):
            try:
                with h5py.File(local_file, 'r') as h:
                    x3d_full = torch.tensor(h['x3d'][:], dtype=self.dtype)
                    x2d_full = torch.tensor(h['x2d'][:], dtype=self.dtype)
                    y_full   = torch.tensor(h['y'][:],   dtype=self.dtype)
                    
                # Subset columns
                x3d = x3d_full[self.col_indices]
                x2d = x2d_full[self.col_indices]
                y   = y_full[self.col_indices]
            except Exception as exc:
                print(f"Error reading {local_file}: {exc}")
                if self.cache_dir:
                    shutil.copy2(filename, local_file)
                continue
            break

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
            iter_end   = min(iter_start + per_worker, len(self.filenames))

        file_indices = list(range(iter_start, iter_end))
        if self.shuffle:
            random.shuffle(file_indices)

        # Return columns from each file
        for idx in file_indices:
            filename = self.filenames[idx]
            x3d, x2d, y = self.read_file(filename)
            # Optionally, you can yield one entire sub-block per file,
            # or yield columns row-by-row as your original code did.
            # Below: yield each column individually
            for x3d_, x2d_, y_ in zip(x3d, x2d, y):
                yield x3d_, x2d_, y_

    def __len__(self):
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
