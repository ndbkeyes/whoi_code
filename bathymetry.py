# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 11:55:38 2021

@author: ndbke
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


# find depth of bottom based on NaNs in vertical profile
def find_bottom(data,depth):
    
    # find & count non-NaN observations
    bath_mask = ~np.isnan(data)               
    bath_count = np.count_nonzero(bath_mask)   
    
    # if location is on land, depth is NaN
    if bath_count == 0:
        bath_depth = np.nan
        
    # get bottom depth by index
    else:
        bath_depth = depth[bath_count-1]
        
    return bath_depth
    

print("opening data file")

# load file for T (using T only, since NaN distribution is same in S)
file = 'C:/Users/ndbke/Dropbox/_NDBK/Research/WHOI/whoi_code/NetCDFs/data.nc'
dat = xr.open_dataset(file, decode_times=False, autoclose=True)


print("building bathymetry")

# construct bathymetry & plot
full_bath = xr.apply_ufunc(find_bottom, dat.temperature, dat.depth, input_core_dims=[["depth"],["depth"]], vectorize=True)
print(full_bath)
full_bath.plot()


print("masking isobath")

# introduce isobath at 200m
lvl = 200
iso_bath = full_bath.where(full_bath >= lvl)
# convert to T/F mask
bath_mask = ~np.isnan(iso_bath)
# apply mask to data (all variables!)
dat = dat.where(bath_mask)

# add bathymetry & mask layers
dat["bathymetry"] = full_bath
dat["bath_mask"] = bath_mask


print("saving & plotting")

# save masked data & bathymetry to NetCDF
dat.to_netcdf('NetCDFs/data_bath.nc',mode='w')

# plot bathymetrically masked datasets (at surface)
plt.figure()
dat.CT.isel(depth=0).plot(robust=True)
plt.figure()
dat.SA.isel(depth=0).plot(robust=True)
plt.figure()
dat.bath_mask.plot()
plt.title(f"{lvl} m isobath")



print("done")