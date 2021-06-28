# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 17:51:55 2021

@author: ndbke
"""

import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt




#%% load and prep datasets

file_t = 'C:/Users/ndbke/Dropbox/_NDBK/Research/WHOI/data/woa18_A5B7_t00_04.nc'
file_s = 'C:/Users/ndbke/Dropbox/_NDBK/Research/WHOI/data/woa18_A5B7_s00_04.nc'
nc_t = xr.open_dataset(file_t, decode_times=False, autoclose=True)
nc_s = xr.open_dataset(file_s, decode_times=False, autoclose=True)

dat_t = nc_t.t_an.isel(time=0,depth=24)
dat_t = dat_t.where(dat_t.lat>60)
dat_t.attrs["long_name"] = "temperature"
dat_s = nc_s.s_an.isel(time=0,depth=24)
dat_s = dat_s.where(dat_s.lat>60)
dat_s.attrs["long_name"] = "salinity"





#%% full T, S datasets in polar view

plt.figure()
pT = dat_t.plot(robust=True,subplot_kws=dict(projection=ccrs.Orthographic(-20, 90), facecolor="white"),transform=ccrs.PlateCarree())
pT.axes.set_global()
pT.axes.coastlines()
pT.axes.set_extent([20,200,20,200])

plt.figure()
pS = dat_s.plot(robust=True,subplot_kws=dict(projection=ccrs.Orthographic(-20, 90), facecolor="white"),transform=ccrs.PlateCarree())
pS.axes.set_global()
pS.axes.coastlines()
pS.axes.set_extent([20,200,20,200])







#%% bathymetry mask in polar view

file_bath = 'NetCDFs/data_isobath.nc'
dat_b = xr.open_dataset(file_bath, decode_times=False, autoclose=True)

print(dat_b)
dat_mask = dat_b.bath_mask.astype(int)
dat_bath = dat_b.bathymetry

plt.figure()
pB = dat_mask.plot(robust=True,subplot_kws=dict(projection=ccrs.Orthographic(-20, 90), facecolor="white"),transform=ccrs.PlateCarree(),cmap="Blues")
pB.axes.set_global()
pB.axes.coastlines()
pB.axes.set_extent([20,200,20,200])

plt.figure()
pBa = dat_bath.plot(robust=True,subplot_kws=dict(projection=ccrs.Orthographic(-20, 90), facecolor="white"),transform=ccrs.PlateCarree(),cmap="viridis")
pBa.axes.set_global()
pBa.axes.coastlines()
pBa.axes.set_extent([20,200,20,200])
