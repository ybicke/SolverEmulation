import math
import shutil
import random
import torch
import h5py
from os.path import join, basename, exists
from torch.utils.data import IterableDataset
from data_utils import get_triangle_indices

class IconIterableDataset3DTendency(IterableDataset):
    """
    Loads triangle data for tendency modeling from each pair of input/output H5 files.
    Each iteration yields one sample: (x3d, x2d, y, w) for the entire triangle at that time.
    """
    def __init__(self,
        triangle_id,
        division_factor,
        input_filenames,
        output_filenames,
        shuffle,
        cache_dir=None,
        dtype='float32',
        total_cols=81920,
    ):
        super().__init__()
        self.input_filenames = input_filenames
        self.output_filenames = output_filenames
        self.cache_dir = cache_dir
        self.shuffle = shuffle
        self.triangle_id = triangle_id
        self.division_factor = division_factor
        
        self.triangle_indices = get_triangle_indices(
            triangle_id, 
            division_factor,
            total_cols
        )
        
        # Convert to numpy for H5 indexing (H5PY doesn't support torch tensors)
        self.triangle_indices_np = self.triangle_indices.numpy()
        
        # Convert dtype string to torch dtype
        self.dtype = getattr(torch, dtype)
        

    def read_file(self, input_filename, output_filename):
        local_input_file = input_filename
        local_output_file = output_filename
        
        if self.cache_dir:
            local_input_file = join(self.cache_dir, basename(input_filename))
            local_output_file = join(self.cache_dir, basename(output_filename))
            
            if not exists(local_input_file):
                shutil.copy2(input_filename, local_input_file)
            if not exists(local_output_file):
                shutil.copy2(output_filename, local_output_file)
        
        # Initialize variables to avoid UnboundLocalError
        x3d = x2d = y = w = temp = None
        
        for _ in range(10):
            try:
                # OPTIMIZED: Read only the triangle indices directly from H5 files
                with h5py.File(local_input_file, 'r') as h_input:
                    # Use fancy indexing to load only the required columns
                    x3d = torch.tensor(h_input['x3d'][self.triangle_indices_np], dtype=self.dtype)
                    x2d = torch.tensor(h_input['x2d'][self.triangle_indices_np], dtype=self.dtype)
                    w = torch.tensor(h_input['w'][self.triangle_indices_np], dtype=self.dtype)
                
                # Read output data - also optimized
                with h5py.File(local_output_file, 'r') as h_output:
                    # Get raw y values - select specific indices and channels in one step
                    y_data = h_output['y'][self.triangle_indices_np]  # Shape: (N_cols, N_levels, N_channels)
                    y = torch.tensor(y_data[:, :, [0, 2, 3, 4, 5, 6, 7]], dtype=self.dtype)
                    temp = torch.tensor(y_data[:, :, 1:2], dtype=self.dtype)
                
                # Concatenate temperature to x3d
                x3d = torch.cat((x3d, temp), dim=-1)
                break  # Success, exit the retry loop
                
            except Exception as exc:
                print(f"Error reading {local_input_file} or {local_output_file}: {exc}")
                if self.cache_dir:
                    shutil.copy2(input_filename, local_input_file)
                    shutil.copy2(output_filename, local_output_file)
                continue
        
        # Check if we successfully loaded the data
        if x3d is None or x2d is None or y is None or w is None:
            raise RuntimeError(f"Failed to load data from {input_filename} and {output_filename} after multiple attempts")
        
        return x3d, x2d, y, w
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self.input_filenames)
        else:
            per_worker = int(math.ceil(len(self.input_filenames) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.input_filenames))
        
        # Create index range for this worker
        file_indices = list(range(iter_start, iter_end))
        if self.shuffle:
            random.shuffle(file_indices)
        
        # Each file pair => 1 sample: the entire triangle
        for idx in file_indices:
            input_filename = self.input_filenames[idx]
            output_filename = self.output_filenames[idx]
            x3d, x2d, y, w = self.read_file(input_filename, output_filename)
            yield x3d, x2d, y, w

    def __len__(self):
        return len(self.input_filenames) 