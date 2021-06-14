# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 11:55:38 2021

@author: ndbke
"""

import numpy as np
import xarray as xr


def find_bottom(data,depth):
    
    bath_mask = ~np.isnan(data)                 # find non-NaN observations
    bath_count = np.count_nonzero(bath_mask)   # count non-NaN observations
    
    if bath_count == 0:
        bath_depth = np.nan
    else:
        bath_depth = depth[bath_count-1]            # find depth of bottom by index
    
    return bath_depth
    

        
        

# filename for T (only using T bc NaN distribution is same in S)
file = 'C:/Users/ndbke/Dropbox/_NDBK/Research/WHOI/WOA_data/woa18_A5B7_t00_04.nc'
dat = xr.open_dataset(file,decode_times=False)

# eliminate time coord, reorder dimensions
dat = dat.isel(time=0)
dat = dat.transpose("lat","lon","depth","nbounds")

### testing on one (lat,lon) profile
# test_profile = dat.t_an.sel(lat=0.125,lon=-60.125)
# test_depth = test_profile.depth
# find_bottom(test_profile.values,test_depth.values)
# bd = xr.apply_ufunc(find_bottom,test_profile,test_depth,input_core_dims=[["depth"],["depth"]])
# print(bd)

# running over full dataset (via vectorization)
full_bath = xr.apply_ufunc(find_bottom, dat.t_an, dat.depth, input_core_dims=[["depth"],["depth"]], vectorize=True)
print(full_bath)
full_bath.plot()
