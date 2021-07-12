# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 18:36:23 2021

@author: ndbke
"""

import xarray as xr
import numpy as np
from utils.plot_tsv import plot_tsv
import matplotlib.pyplot as plt

file_grn = 'NetCDFs/tsv_grn.nc'
dat_grn = xr.open_dataset(file_grn, decode_times=False, autoclose=True)
file_cmk = 'NetCDFs/tsv_cmk.nc'
dat_cmk = xr.open_dataset(file_cmk, decode_times=False, autoclose=True)

# ratio the two! and take out any places where they're both zero
dv = ( np.nan_to_num(dat_cmk.volume.values)+1 ) / ( np.nan_to_num(dat_grn.volume.values)+1 )
dv[dv == 1] = 0
diff = xr.DataArray( dv , coords=[dat_cmk.t,dat_cmk.s], dims=["t","s"] )

# plot log-scaled ratios
plot_tsv(diff,norm="log_signed")
plt.title("Ratio between Carmack and WOA vol T-S values")

dat_grn.close()
dat_cmk.close()
