import math
import shutil
from os.path import join, basename, exists

import h5py
import torch
import random
from torch.utils.data import IterableDataset, DataLoader
import torch.nn.functional as F


class IconColumnIterableDataset(IterableDataset):
    def __init__(self, input_filenames, output_filenames, shuffle=None, subsample=None, dtype='float32', cache_dir=None):
        super(IconColumnIterableDataset).__init__()
        self.input_filenames = input_filenames
        self.output_filenames = output_filenames
        self.dtype = torch.float32
        self.cache_dir = cache_dir
        self.shuffle = shuffle
        self.subsample = subsample
        
        
        """
        # 3D features (x3d)
                    ['u - Zonal wind [m/s]',
                    'v - Meridional wind [m/s]',
                    'geopot - Geopotential [m²/s²]',
                    'pres - Pressure [hPa]',
                    'clc - Cloud cover fraction [-]']
                    'qc - Cloud water content [kg/kg]',
                    'qi - Cloud ice content [kg/kg]',
                    'qv - Water vapor specific humidity [kg/kg]',
                    --- 'temp' added in the end
                    --- 'w' added later
                        

        # 2D features (x2d)
        ['pres_sfc - Surface pressure [hPa]', 
        'cosmu0 - Cosine of solar zenith angle [-]', 
        'qv_s - Surface water vapor specific humidity [kg/kg]']
        
        # y values        
        ['ddt_temp_sum',   # sum of temperature tendencies
        'temp',            # Temperature
        'ddt_temp_dyn',    # dynamical temperature tendency
        'ddt_u_sum',       # sum of zonal wind tendencies
        'ddt_v_sum',       # sum of meridional wind tendencies
        'ddt_qv_conv',     # convective tendency of absolute humidity
        'ddt_qc_conv',     # convective tendency of cloud water mass density
        'ddt_qi_conv',     # convective tendency of cloud ice mass density<<
    ] 
        """            

    def read_file(self, input_filename, output_filename):
        local_input_file = join(self.cache_dir, basename(input_filename))
        local_output_file = join(self.cache_dir, basename(output_filename))

        if not exists(local_input_file) and self.cache_dir:
            shutil.copy2(input_filename, local_input_file)
        if not exists(local_output_file) and self.cache_dir:
            shutil.copy2(output_filename, local_output_file)            

        x3d = None
        x2d = None
        y = None

        for _ in range(10):
            try:
                with h5py.File(local_input_file, 'r') as h_input:
                    x3d = torch.tensor(h_input['x3d'][:, :, :], dtype=self.dtype)
                    x2d = torch.tensor(h_input['x2d'][:], dtype=self.dtype)
                    w = torch.tensor(h_input['w'][:], dtype=self.dtype)
                                       

                with h5py.File(local_output_file, 'r') as h_output:
                    
                    # Scale input features based on their units
                    seconds_in_3hours = 3 * 60 * 60  # 10800 seconds
                    
                    # Get raw y values
                    y = torch.tensor(h_output['y'][:, :, [0, 2, 3, 4, 5, 6, 7]], dtype=self.dtype)
                    temp = torch.tensor(h_output['y'][:, :, 1:2], dtype=self.dtype)
                    
                    # Scale each tendency based on its units
                    # Order: [ddt_temp_sum, ddt_temp_dyn, ddt_u_sum, ddt_v_sum, ddt_qv_conv, ddt_qc_conv, ddt_qi_conv]
                    y_scaling_factors = torch.ones(7, dtype=self.dtype)
                    y_scaling_factors[0:2] *= seconds_in_3hours    # Temperature tendencies: K s^-1 -> K/(3h)
                    y_scaling_factors[2:4] *= (seconds_in_3hours ** 2)    # Wind tendencies: m s^-2 -> m/(3h)^2
                    y_scaling_factors[4:] *= seconds_in_3hours    # Mass density tendencies: kg m^-3 s^-1 -> kg m^-3/(3h)
                    
                    
                    # Apply scaling factors to y
                    y = y * y_scaling_factors.to(y.device)

            except OSError as e:
                print(f"Error reading files: {e}. Retrying...")
                shutil.copy2(input_filename, local_input_file)
                shutil.copy2(output_filename, local_output_file)
                continue
            break

        # Check if data was successfully read
        if x3d is None or x2d is None or y is None:
            raise ValueError(f"Failed to read data from files after retries: {input_filename}, {output_filename}")
        
        x3d = torch.cat((x3d, temp), dim=-1)

        if self.subsample:
            indices = torch.randint(0, y.size(0), (int(self.subsample * y.size(0)),))
            return x3d[indices], x2d[indices], y[indices], w[indices]

        return x3d, x2d, y, w
    

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = 0
            iter_end = len(self.input_filenames)
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil(len(self.input_filenames) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.input_filenames))

        if self.shuffle:
            # Shuffle the filenames within the worker's range
            indices = list(range(iter_start, iter_end))
            random.shuffle(indices)
            input_filenames = [self.input_filenames[i] for i in indices]
            output_filenames = [self.output_filenames[i] for i in indices]
        else:
            input_filenames = self.input_filenames[iter_start:iter_end]
            output_filenames = self.output_filenames[iter_start:iter_end]

        for input_filename, output_filename in zip(input_filenames, output_filenames):
            x3d, x2d, y, w = self.read_file(input_filename, output_filename)
            for x3d_, x2d_, y_, w in zip(x3d, x2d, y, w):
                yield x3d_, x2d_, y_, w