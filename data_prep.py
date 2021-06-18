# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 16:41:52 2021

@author: ndbke
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import gsw



print("opening data files")

# load files for T and S
file_t = 'C:/Users/ndbke/Dropbox/_NDBK/Research/WHOI/WOA_data/woa18_A5B7_t00_04.nc'
file_s = 'C:/Users/ndbke/Dropbox/_NDBK/Research/WHOI/WOA_data/woa18_A5B7_s00_04.nc'
nc_t = xr.open_dataset(file_t, decode_times=False, autoclose=True)
nc_s = xr.open_dataset(file_s, decode_times=False, autoclose=True)

# get correct DataArrays
dat_t = nc_t.t_an
dat_s = nc_s.s_an


print("filtering to Arctic Ocean")

# filter data to Arctic Ocean at t=0
dat_t = dat_t.isel(time=0)
dat_t = dat_t.where(dat_t.lat > 60)
dat_s = dat_s.isel(time=0)
dat_s = dat_s.where(dat_t.lat > 60)

# print resulting arrays
print(dat_t)
print(dat_s)


print("making DataSet object")

# make dataset with T, S for Arctic Ocean only
dataset = xr.Dataset(
    {
        "temperature": dat_t,
        "salinity": dat_s
    },
    coords={
        "lat": dat_t.coords["lat"],
        "lon": dat_t.coords["lon"],
        "depth": dat_t.coords["depth"]
    }
)

print("converting to CT, SA")

# add conservative temperature & absolute salinity & assoc quantities to dataset
dataset['pressure'] = xr.apply_ufunc( gsw.p_from_z, -dataset.depth, np.mean(dataset.lat) )
dataset['SA']       = xr.apply_ufunc( gsw.SA_from_SP, dataset.salinity, dataset.pressure, dataset.lon, dataset.lat )
dataset['pot_temp'] = xr.apply_ufunc( gsw.pt0_from_t, dataset.SA, dataset.temperature, dataset.pressure )
dataset['CT']       = xr.apply_ufunc( gsw.CT_from_pt, dataset.SA, dataset.pot_temp )
dataset['pot_dens'] = xr.apply_ufunc( gsw.sigma0, dataset.SA, dataset.CT )


print("saving & plotting")


# print & save to file
print(dataset)
dataset.to_netcdf("NetCDFs/data.nc",mode='w')


# plot both datasets (at surface)
plt.figure()
dataset.CT.isel(depth=0).plot(robust=True)
plt.ylim(60,90)
plt.figure()
dataset.SA.isel(depth=66).plot(robust=True)
plt.ylim(60,90)


print("done")