# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 17:21:26 2021

@author: ndbke
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from utils.plot_tsv import plot_tsv
from utils.entropy import entropy_all
    
    
    
#%%

# read in CSV file
cmk_data = np.genfromtxt("../data/tsv_grn_summer.csv", delimiter=',')

# get T and S coordinates
T_bins = cmk_data[1:,0]
S_bins = cmk_data[0,1:]

# trim T, S coordinate col/row out of data matrix
cmk_data = cmk_data[1:,1:]


# save as DataArray & NetCDF
tsv_cmk = xr.DataArray( cmk_data , coords=[T_bins,S_bins], dims=["t","s"])
tsv_cmk.name = "volume"
tsv_cmk.to_netcdf("NetCDFs/tsv_cmk.nc")
tsv_cmk.close()

# plot volumetric T-S
plt.figure()
plot_tsv(tsv_cmk,xylabels=["salinity","potential temperature"])



#%% select different water masses

# make and apply condition on T, S coords
cond = (tsv_cmk.t >= -1.5) & (tsv_cmk.t < 0) & (tsv_cmk.s >= 34.85) & (tsv_cmk.s < 34.95)
deep = tsv_cmk.where(cond)
upper = tsv_cmk.where(~cond)

# plot upper and lower water masses on T-S
plt.figure()
plot_tsv(upper,xylabels=["salinity","potential temperature"])
plt.figure()
plot_tsv(deep,xylabels=["salinity","potential temperature"])




#%% entropy calculations

entropy_all(tsv_cmk,disp=True)
