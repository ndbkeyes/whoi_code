# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 12:00:45 2021

@author: ndbke
"""


import xarray as xr
import numpy as np
# import regionmask as rgm
import matplotlib.pyplot as plt
from utils.make_tsv import make_tsv
from utils.entropy import entropy_all
import cartopy.crs as ccrs


print("opening & filtering files")

file_data = 'NetCDFs/data_all.nc'
dat_arc = xr.open_dataset(file_data, decode_times=False, autoclose=True)
file_bath = 'NetCDFs/bathymetry.nc'
bath = xr.open_dataset(file_bath, decode_times=False, autoclose=True)
file_vol = 'NetCDFs/volume.nc'
vol = xr.open_dataset(file_vol, decode_times=False, autoclose=True)
dat_arc.close()
vol.close()



#%% filter & mask geographically


# filter to Arctic Circle and to 200m isobath
dat_arc = dat_arc.where(dat_arc.lat >= 60)
dat_arc = dat_arc.where(bath.bath_mask)



# rightmost part of Pacific - near Russia
mask1 = (dat_arc.lat > 60) & (dat_arc.lat < 65) & (dat_arc.lon > 150)  & (dat_arc.lon < 180)
# leftmost part of Pacific - near Alaska
mask2 = (dat_arc.lat > 60) & (dat_arc.lat < 65) & (dat_arc.lon > -180) & (dat_arc.lon < -160)
# inland area by Nordic countries
mask3 = (dat_arc.lat > 60) & (dat_arc.lat < 67) & (dat_arc.lon > 17)   & (dat_arc.lon < 40)
# Canadian island water & Hudson Bay
mask4 = (dat_arc.lat > 60) & (dat_arc.lat < 72) & (dat_arc.lon > -130) & (dat_arc.lon < -70)
# Norwegian coastal area
mask5 = (dat_arc.lat > 60) & (dat_arc.lat < 72) & (dat_arc.lon > 2) & (dat_arc.lon < 40)



# # custom polygon  for Norwegian coast - using regionmask IN PROGRESS TODO ***
# mask5 = np.array([[60,7],[60,2],[72,20],[72,27]])
# rmask = rgm.Regions([mask5])
# rmask.plot()



# plot masked areas over map
mask_plot = mask1 | mask2 | mask3 | mask4 | mask5
plt.figure()
globe = dat_arc.pot_temp.isel(depth=0).plot(robust=True,subplot_kws=dict(projection=ccrs.Orthographic(-20, 90), facecolor="white"),transform=ccrs.PlateCarree())
mask_plot = mask_plot.where(mask_plot == 1)
mask_plot.plot(alpha=0.025,ax=globe.axes,transform=ccrs.PlateCarree())
globe.axes.set_global()
globe.axes.coastlines()
globe.axes.set_extent([20,200,20,200])



# apply regional masks
cond = (~mask1) & (~mask2) & (~mask3) & (~mask4) & (~mask5)
dat_arc = dat_arc.where( cond )
dat_arc = dat_arc.where( cond )




#%% make TSV out of the filtered data & do entropy

tsv_arc = make_tsv(dat_arc,vol,res=[0.1,0.05],tsbounds=[-3,12,22,36],name="arc",convert=False)

entropy_all(tsv_arc,disp=True)


#%% plot individual profiles

lat_lower = 60.125
lat_upper = 88.125
num_pts = (lat_upper - lat_lower) / 0.25 + 1

lat_arr = np.linspace(lat_lower,lat_upper,int(num_pts))
lon_arr = np.repeat(0.125,len(lat_arr))
coord_arr = np.array(list(zip(lat_arr,lon_arr)))

plt.figure()
for i in range(len(coord_arr)):
    print(coord_arr[i])
    prof = dat_arc.sel(lat=coord_arr[i,0],lon=coord_arr[i,1])
    if coord_arr[i,0] < 75:
        col = "blue"
    else:
        col = "red"
    plt.plot(prof.salinity, prof.temperature, color=col)

plt.xlabel("SA")
plt.ylabel("CT")
plt.title("Individual Arctic T-S profiles along 0 deg. longitude")
