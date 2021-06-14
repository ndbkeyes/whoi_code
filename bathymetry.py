# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 11:55:38 2021

@author: ndbke
"""

import numpy as np
import xarray as xr


# find depth of bottom based on NaNs in vertical profile
def find_bottom(data,depth,isobath):
    
    # find & count non-NaN observations
    bath_mask = ~np.isnan(data)               
    bath_count = np.count_nonzero(bath_mask)   
    
    # if location is on land, depth is NaN
    if bath_count == 0:
        bath_depth = np.nan
        
    else:
        
        # get bottom depth by index
        bath_depth = depth[bath_count-1]
        
        # if above given isobath, set to NaN
        if bath_depth < isobath:
            bath_depth = np.nan
    
    return bath_depth
    

# load file for T (using T only, since NaN distribution is same in S)
file = 'C:/Users/ndbke/Dropbox/_NDBK/Research/WHOI/WOA_data/woa18_A5B7_t00_04.nc'
dat = xr.open_dataset(file,decode_times=False)

# eliminate time coord, reorder dimensions
dat = dat.isel(time=0)
dat = dat.transpose("lat","lon","depth","nbounds")

# running over full dataset (via vectorization)
full_bath = xr.apply_ufunc(find_bottom, dat.t_an, dat.depth, 0, input_core_dims=[["depth"],["depth"],[]], vectorize=True)
print(full_bath)
full_bath.plot()

# add bathymetry to TSV NetCDF file
file_tsv = 'C:/Users/ndbke/Dropbox/_NDBK/Research/WHOI/whoi_code/tsv.nc'
tsv = xr.open_dataset(file_tsv, decode_times=False)
tsv["bathymetry"] = full_bath
tsv.to_netcdf("tsv.nc")
tsv.close()