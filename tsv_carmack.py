# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 17:21:26 2021

@author: ndbke
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from utils.plot_tsv import plot_tsv
    
    
    
#%%

# read in CSV file
grn_tsv = np.genfromtxt("../data/tsv_grn_summer.csv", delimiter=',')

# get T and S coordinates
T_bins = grn_tsv[1:,0]
S_bins = grn_tsv[0,1:]

print(T_bins)
print(S_bins)

# trim T, S coordinates out of data matrix
grn_tsv = grn_tsv[1:,1:]


# save as DataArray & NetCDF
tsv = xr.DataArray( grn_tsv , coords=[T_bins,S_bins], dims=["temperature","salinity"])
tsv.name = "volume"
tsv.to_netcdf("NetCDFs/tsv_grn.nc")
tsv.close()

# plot volumetric T-S
plt.figure()
plot_tsv(tsv,xylabels=["salinity","potential temperature"])



#%% select different water masses

# make and apply condition on T, S coords
cond = (tsv.temperature >= -1.5) & (tsv.temperature < 0) & (tsv.salinity >= 34.85) & (tsv.salinity < 34.95)
deep = tsv.where(cond)
upper = tsv.where(~cond)

# plot upper and lower water masses on T-S
plt.figure()
plot_tsv(upper,xylabels=["salinity","potential temperature"])
plt.figure()
plot_tsv(deep,xylabels=["salinity","potential temperature"])
