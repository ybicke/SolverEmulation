import math
import random
import torch
import h5py
from torch.utils.data import IterableDataset
from utils.data_utils import get_triangle_indices


class TendencyDataset(IterableDataset):
    """
    Simplified unified dataset loader for tendency training.
    No caching - reads directly from network storage for reliability.
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
        total_cols=81920,
        use_lonlat=False,  # New parameter for lon/lat features
        grid_file_path=None,  # Required when use_lonlat=True
    ):
        self.input_filenames = input_filenames
        self.output_filenames = output_filenames
        self.shuffle = shuffle
        self.mode = mode
        self.subsample = subsample
        self.use_lonlat = use_lonlat
        self.grid_file_path = grid_file_path
        
        # Validate lon/lat usage requirements
        if self.use_lonlat and not grid_file_path:
            raise ValueError("grid_file_path is required when use_lonlat=True")
        
        # Set up triangle indices based on dataset type
        if dataset_type == 'triangle':
            self.triangle_indices = get_triangle_indices(
                triangle_id, 
                division_factor=division_factor,
                total_cols=total_cols
            ).numpy()  # Convert to numpy for H5 indexing
            
            print(f"Triangle {triangle_id} with division_factor {division_factor}: "
                  f"{len(self.triangle_indices)} columns")
            
        else:
            self.triangle_indices = None
            print(f"Full dataset mode: loading all {total_cols} columns")
        
        # Initialize lon/lat coordinates and normalization if requested
        self.lonlat_coords = None
        self.lonlat_mean = None
        self.lonlat_std = None
        
        if self.use_lonlat:
            self._initialize_lonlat_features()
        

    def _initialize_lonlat_features(self):
        """Initialize longitude/latitude coordinates and normalization."""
        from utils.data_utils import get_lonlat_coordinates_for_triangle, create_lonlat_normalizer, normalize_lonlat_coordinates
        
        if self.triangle_indices is not None:
            # Triangle mode: get coordinates for triangle area
            triangle_indices_tensor = torch.tensor(self.triangle_indices, dtype=torch.long)
            lon_coords, lat_coords = get_lonlat_coordinates_for_triangle(
                self.grid_file_path, triangle_indices_tensor, device='cpu'
            )
            
            # Create normalization statistics for this triangle
            self.lonlat_mean, self.lonlat_std = create_lonlat_normalizer(
                self.grid_file_path, [triangle_indices_tensor]
            )
            
            # Normalize coordinates
            self.lonlat_coords = normalize_lonlat_coordinates(
                lon_coords, lat_coords, self.lonlat_mean, self.lonlat_std
            )
            print(f"Initialized lon/lat coordinates for triangle: shape {self.lonlat_coords.shape}")
            print(f"Lon/lat normalization - mean: {self.lonlat_mean}, std: {self.lonlat_std}")
        else:
            # Full mode: would need coordinates for all cells
            raise NotImplementedError("Lon/lat features not yet implemented for full dataset mode")
    
    def get_lonlat_features(self, x2d_shape):
        """
        Get normalized lon/lat coordinates to be added after normalization.
        Args:
            x2d_shape: Shape of the x2d tensor to match
        Returns:
            Normalized lon/lat coordinates with matching shape
        """
        if not self.use_lonlat or self.lonlat_coords is None:
            return None
        
        # Handle different x2d shapes based on mode and batch size
        if len(x2d_shape) == 2:
            # x2d is [batch_size, features] - 1D mode batched
            batch_size = x2d_shape[0]
            if batch_size == len(self.lonlat_coords):
                # Each batch item is a different column
                return self.lonlat_coords
            else:
                raise ValueError(f"Batch size {batch_size} doesn't match number of columns {len(self.lonlat_coords)}")
        
        elif len(x2d_shape) == 3:
            # x2d is [batch_size, num_columns, features] - 3D mode batched
            batch_size, num_columns = x2d_shape[0], x2d_shape[1]
            if num_columns == len(self.lonlat_coords):
                # Expand lonlat_coords for batch dimension
                lonlat_batched = self.lonlat_coords.unsqueeze(0).expand(batch_size, -1, -1)
                return lonlat_batched
            else:
                raise ValueError(f"Number of columns {num_columns} doesn't match lon/lat coords {len(self.lonlat_coords)}")       
        else:
            raise ValueError(f"Unexpected x2d shape: {x2d_shape}")

    def read_file_pair(self, input_filename, output_filename):
        """Read and preprocess H5 files directly from network storage"""
        # Read files directly - no caching, no race conditions
        with h5py.File(input_filename, 'r') as h_input, h5py.File(output_filename, 'r') as h_output:
            # Load data based on triangle mode
            if self.triangle_indices is not None:
                # Triangle mode: load subset
                x3d = torch.tensor(h_input['x3d'][self.triangle_indices], dtype=torch.float32)
                x2d = torch.tensor(h_input['x2d'][self.triangle_indices], dtype=torch.float32)
                w = torch.tensor(h_input['w'][self.triangle_indices], dtype=torch.float32)
                
                # Load outputs for triangle
                y_data = h_output['y'][self.triangle_indices]
                y = torch.tensor(y_data[:, :, [0, 2, 3, 4, 5, 6, 7]], dtype=torch.float32)
                temp = torch.tensor(y_data[:, :, 1:2], dtype=torch.float32)
            else:
                # Full mode: load everything
                x3d = torch.tensor(h_input['x3d'][:], dtype=torch.float32)
                x2d = torch.tensor(h_input['x2d'][:], dtype=torch.float32)
                w = torch.tensor(h_input['w'][:], dtype=torch.float32)
                
                # Load all outputs
                y_data = h_output['y'][:]
                y = torch.tensor(y_data[:, :, [0, 2, 3, 4, 5, 6, 7]], dtype=torch.float32)
                temp = torch.tensor(y_data[:, :, 1:2], dtype=torch.float32)
            
            # Concatenate temperature to x3d
            x3d = torch.cat([x3d, temp], dim=-1)
            
            # Apply subsampling if specified
            if self.subsample is not None and self.subsample < 1.0:
                num_samples = int(self.subsample * x3d.shape[0])
                if num_samples > 0:
                    indices = torch.randperm(x3d.shape[0])[:num_samples]
                    x3d, x2d, y, w = x3d[indices], x2d[indices], y[indices], w[indices]
        
            
            return x3d, x2d, y, w

    def __iter__(self):
        """Simple iterator with optional shuffling"""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading
            iter_start = 0
            iter_end = len(self.input_filenames)
        else:  # in a worker process - split workload
            per_worker = int(math.ceil(len(self.input_filenames) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.input_filenames))
        
        # Create file index list for this worker
        file_indices = list(range(iter_start, iter_end))
        
        # Shuffle if requested (like your original simple loader)
        if self.shuffle:
            random.shuffle(file_indices)
        
        # Process files in the (potentially shuffled) order
        for i in file_indices:
            input_filename = self.input_filenames[i]
            output_filename = self.output_filenames[i]
            x3d, x2d, y, w = self.read_file_pair(input_filename, output_filename)
            
            if self.mode == '1d':
                # Column-wise: yield each column separately (like original x3d_, x2d_, y_)
                for x3d_, x2d_, y_, w_ in zip(x3d, x2d, y, w):
                    yield x3d_, x2d_, y_, w_
            elif self.mode == '3d':
                # Triangle-wise: yield entire triangle
                yield x3d, x2d, y, w


# Backward compatibility
IconColumnDataset_Tendency = TendencyDataset
IconIterableDataset3DTendency = TendencyDataset 