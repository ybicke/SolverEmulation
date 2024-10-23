import math
import shutil
from os.path import join, basename, exists

import h5py
import torch
from torch.utils.data import IterableDataset, DataLoader


class IconColumnIterableDataset(IterableDataset):
    def __init__(self, filenames, neighbourhoods, shuffle=None, subsample=None, dtype='flaot32', cache_dir=None):
        super(IconColumnIterableDataset).__init__()
        self.filenames = filenames
        self.neighbourhoods = neighbourhoods
        self.dtype = torch.float32
        self.cache_dir = cache_dir
        self.shuffle = shuffle
        self.subsample = subsample

    def read_file(self, filename):
        local_file = join(self.cache_dir, basename(filename))
        if not exists(local_file) and self.cache_dir:
            shutil.copy2(filename, local_file)
        for _ in range(10):
            try:
                with h5py.File(local_file, 'r') as h:
                    x3d = torch.tensor(h['x3d'][:], dtype=self.dtype)
                    x2d = torch.tensor(h['x2d'][:], dtype=self.dtype)
                    y = torch.tensor(h['y'][:], dtype=self.dtype)
            except:
                shutil.copy2(filename, local_file)
                continue
            break         

        num_cells = x2d.shape[0]
        assert num_cells == self.neighborhoods.shape[0], "Mismatch in number of cells and neighborhoods"


        if self.subsample:
            total_samples = int(self.subsample * len(self.neighborhoods))
            indices = torch.randint(0, len(self.neighborhoods), (total_samples,))
            neighborhoods = self.neighborhoods[indices]
        else:
            neighborhoods = self.neighborhoods
            
        # Flatten the neighborhood indices to index into x2d and x3d
        flat_indices = neighborhoods.flatten()

        # Gather data
        x2d_neighborhood = x2d[flat_indices].view(-1, neighborhoods.shape[1], x2d.size(1))
        x3d_neighborhood = x3d[flat_indices].view(-1, neighborhoods.shape[1], x3d.size(1), x3d.size(2))
        y_neighborhood = y[flat_indices].view(-1, neighborhoods.shape[1], y.size(1))

        return x3d_neighborhood, x2d_neighborhood, y_neighborhood

    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = 0
            iter_end = len(self.filenames)
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil(len(self.filenames) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.filenames))
            
        if self.shuffle:
            # Shuffle the filenames within the worker's range   
            indices = list(range(iter_start, iter_end))
            random.shuffle(indices)
            filenames = [self.filenames[i] for i in indices]
        else:
            filenames = self.filenames[iter_start:iter_end]
            
            
        for filename in filenames:
            x3d, x2d, y = self.read_file(filename)
            for x3d_, x2d_, y_ in zip(x3d, x2d, y):
                yield x3d_, x2d_, y_

