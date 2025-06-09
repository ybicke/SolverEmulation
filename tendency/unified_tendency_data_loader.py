import math
import shutil
import random
import torch
import h5py
from os.path import join, basename, exists
from torch.utils.data import IterableDataset
from data_utils import get_triangle_indices


class UnifiedTendencyDataset(IterableDataset):
    """
    Unified dataset loader for tendency training that supports:
    - Both 1D (column-wise) and 3D (triangle-wise) modes
    - Both triangle and full dataset selection
    - Consistent data preprocessing for both modes
    """
    def __init__(
        self,
        input_filenames,
        output_filenames,
        mode='1d',  # '1d' for column-wise, '3d' for triangle-wise
        dataset_type='triangle',  # 'triangle' or 'full'
        triangle_id=39,
        division_factor=4,
        shuffle=False,
        subsample=None,
        dtype='float32',
        cache_dir=None,
        total_cols=81920,
    ):
        super().__init__()
        self.input_filenames = input_filenames
        self.output_filenames = output_filenames
        self.cache_dir = cache_dir
        self.shuffle = shuffle
        self.mode = mode
        self.dataset_type = dataset_type
        self.triangle_id = triangle_id
        self.division_factor = division_factor
        self.subsample = subsample
        
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
        
        print(f"Mode: {mode} ({'column-wise' if mode == '1d' else 'triangle-wise'} sampling)")

    def read_file(self, input_filename, output_filename):
        """Read and preprocess H5 files, returning data from selected area"""
        local_input_file = input_filename
        local_output_file = output_filename
        
        # Optimize caching - only copy if not already cached
        if self.cache_dir:
            local_input_file = join(self.cache_dir, basename(input_filename))
            local_output_file = join(self.cache_dir, basename(output_filename))
            
            # Check if both files need copying before starting
            need_input_copy = not exists(local_input_file)
            need_output_copy = not exists(local_output_file)
            
            if need_input_copy:
                shutil.copy2(input_filename, local_input_file)
            if need_output_copy:
                shutil.copy2(output_filename, local_output_file)
        
        for _ in range(10):  # Try up to 10 times if there are file access issues
            try:
                with h5py.File(local_input_file, 'r') as h_input:
                    if self.is_triangle_mode:
                        # Load only triangle columns to save memory and time
                        x3d = torch.tensor(h_input['x3d'][self.triangle_indices.numpy(), :, :], dtype=self.dtype)
                        x2d = torch.tensor(h_input['x2d'][self.triangle_indices.numpy()], dtype=self.dtype)
                        w = torch.tensor(h_input['w'][self.triangle_indices.numpy()], dtype=self.dtype)
                    else:
                        # Load all columns
                        x3d = torch.tensor(h_input['x3d'][:, :, :], dtype=self.dtype)
                        x2d = torch.tensor(h_input['x2d'][:], dtype=self.dtype)
                        w = torch.tensor(h_input['w'][:], dtype=self.dtype)

                with h5py.File(local_output_file, 'r') as h_output:
                    if self.is_triangle_mode:
                        # Load only triangle columns for specific channels (exclude temp at index 1)
                        y_data = h_output['y'][self.triangle_indices.numpy()]
                        y = torch.tensor(y_data[:, :, [0, 2, 3, 4, 5, 6, 7]], dtype=self.dtype)
                        temp = torch.tensor(y_data[:, :, 1:2], dtype=self.dtype)
                    else:
                        # Load all columns for specific channels (exclude temp at index 1)
                        y_data = h_output['y'][:]
                        y = torch.tensor(y_data[:, :, [0, 2, 3, 4, 5, 6, 7]], dtype=self.dtype)
                        temp = torch.tensor(y_data[:, :, 1:2], dtype=self.dtype)
                
                # Concatenate temperature to x3d
                x3d = torch.cat((x3d, temp), dim=-1)
                
                # Apply subsampling if specified
                if self.subsample is not None and self.subsample < 1.0:
                    num_samples = int(self.subsample * x3d.shape[0])
                    if num_samples > 0:
                        indices = torch.randperm(x3d.shape[0])[:num_samples]
                        x3d = x3d[indices]
                        x2d = x2d[indices]
                        y = y[indices]
                        w = w[indices]
                
                return x3d, x2d, y, w
                
            except Exception as exc:
                print(f"Error reading {local_input_file} or {local_output_file}: {exc}")
                # Only re-copy on error, not every retry
                if self.cache_dir:
                    shutil.copy2(input_filename, local_input_file)
                    shutil.copy2(output_filename, local_output_file)
                continue
            break
            
        # If we reach here, all attempts failed
        raise RuntimeError(f"Failed to read files {input_filename} and {output_filename} after multiple attempts")
    
    def __iter__(self):
        """Iterator that yields data based on mode (1D or 3D)"""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self.input_filenames)
        else:
            per_worker = int(math.ceil(len(self.input_filenames) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.input_filenames))
        
        file_indices = list(range(iter_start, iter_end))
       
        if self.shuffle:
            random.shuffle(file_indices)
        
        # Process each file
        for idx in file_indices:
            input_filename = self.input_filenames[idx]
            output_filename = self.output_filenames[idx]
            x3d, x2d, y, w = self.read_file(input_filename, output_filename)
            
            if self.mode == '1d':
                # 1D mode: Return each column as a separate sample
                for col_idx in range(len(x3d)):
                    yield x3d[col_idx], x2d[col_idx], y[col_idx], w[col_idx]
            elif self.mode == '3d':
                # 3D mode: Return entire triangle as one sample
                yield x3d, x2d, y, w
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
            return len(self.input_filenames) * approx_columns_per_file
        elif self.mode == '3d':
            # 3D mode: Each file is one sample
            return len(self.input_filenames)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


# Backward compatibility aliases
IconColumnDataset_Tendency = UnifiedTendencyDataset
IconIterableDataset3DTendency = UnifiedTendencyDataset 