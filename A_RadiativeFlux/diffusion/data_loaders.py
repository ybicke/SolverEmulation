import math
import shutil
from os.path import join, basename, exists

import h5py
import torch
from torch.utils.data import IterableDataset


class IconColumnIterableDataset(IterableDataset):
    def __init__(self, filenames, shuffle=None, subsample=None, dtype='flaot32', cache_dir=None):
        super(IconColumnIterableDataset).__init__()
        self.filenames = filenames
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
                    # x3d = torch.flip(x3d, dims=[1])
                    # y = torch.flip(y, dims=[1])

            except:
                shutil.copy2(filename, local_file)
                continue
            break         
        if self.shuffle:
            # TODO: Add shuffle        
            pass

        if self.subsample:
            indices = torch.randint(0, y.size(0), (int(self.subsample*y.size(0)), ))
            return x3d[indices], x2d[indices], y[indices]

        return x3d, x2d, y
    
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
            
        for filename in self.filenames[iter_start: iter_end]:
            x3d, x2d, y = self.read_file(filename)
            for x3d_, x2d_, y_ in zip(x3d, x2d, y):
                yield x3d_, x2d_, y_

