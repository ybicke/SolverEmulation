import glob, os
import numpy as np
from netCDF4 import Dataset


def generate_icon_sample_dataset(fname_out="./sample_icon_1day.nc"):
    ## Use last 8 indices for sample online dataset
    fl = np.sort(
        glob.glob(
            os.path.join(
                "/net/ch4/atmcirc/gbertoli/ICON_output/latest",
                "APE_JABW_R02B05_G_ECRAD_atm_3d_DOM01_ml_00*_lonlat.nc",
            )
        )
    ).tolist()
    fn = fl[-1]
    num_time_samples = 8
    with Dataset(fname_out, mode="w", format="NETCDF4") as fh:
        with Dataset(fn, mode="r") as fh_read:
            ## global attribs
            fh.setncatts(fh_read.__dict__)
            ## dims
            for k, dimension in fh_read.dimensions.items():
                fh.createDimension(
                    k, (len(dimension) if not dimension.isunlimited() else None)
                )
            for k, variable in fh_read.variables.items():
                variable = fh_read[k]
                fh.createVariable(k, variable.datatype, variable.dimensions)
                ## copy variable attributes
                fh[k].setncatts(fh_read[k].__dict__)
                ## copy content
                if "time" in variable.dimensions:
                    fh[k][:] = fh_read[k][-num_time_samples:]
                else:
                    fh[k][:] = fh_read[k][:]
