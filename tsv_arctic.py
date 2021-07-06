# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 12:00:45 2021

@author: ndbke
"""


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from utils.make_tsv import make_tsv



#%% mask data & make TSV

print("opening & filtering files")

file_data = 'NetCDFs/data_all.nc'
dat = xr.open_dataset(file_data, decode_times=False, autoclose=True)
file_bath = 'NetCDFs/bathymetry.nc'
bath = xr.open_dataset(file_bath, decode_times=False, autoclose=True)
file_vol = 'NetCDFs/volume.nc'
vol = xr.open_dataset(file_vol, decode_times=False, autoclose=True)


# filter to Arctic Circle and to 200m isobath
dat = dat.where(dat.lat >= 60)
dat = dat.where(bath.bath_mask)


# rightmost part of Pacific - near Russia
mask1 = (dat.lat > 60) & (dat.lat < 65) & (dat.lon > 150)  & (dat.lon < 180)
# leftmost part of Pacific - near Alaska
mask2 = (dat.lat > 60) & (dat.lat < 65) & (dat.lon > -180) & (dat.lon < -160)
# areas by Nordic countries
mask3 = (dat.lat > 60) & (dat.lat < 67) & (dat.lon > 17)   & (dat.lon < 40)
# Canadian island water & Hudson Bay
mask4 = (dat.lat > 60) & (dat.lat < 72) & (dat.lon > -130) & (dat.lon < -70)


# filter out data in masking areas
cond = (~mask1) & (~mask2) & (~mask3) & (~mask4)
dat["CT"] = dat.CT.where( cond )
dat["SA"] = dat.SA.where( cond )

dat.close()
vol.close()


# make TSV out of the filtered data
make_tsv(dat,vol,res=[0.1,0.05],tsbounds=[-2,12,22,36],name="arc")




#%% plot profiles from different locations

lat_lower = 60.125
lat_upper = 88.125
num_pts = (lat_upper - lat_lower) / 0.25 + 1

lat_arr = np.linspace(lat_lower,lat_upper,int(num_pts))
lon_arr = np.repeat(0.125,len(lat_arr))
coord_arr = np.array(list(zip(lat_arr,lon_arr)))

plt.figure()
for i in range(len(coord_arr)):
    print(coord_arr[i])
    prof = dat.sel(lat=coord_arr[i,0],lon=coord_arr[i,1])
    if coord_arr[i,0] < 75:
        col = "blue"
    else:
        col = "red"
    plt.plot(prof.salinity, prof.temperature, color=col)

plt.xlabel("SA")
plt.ylabel("CT")
plt.title("Individual Arctic T-S profiles along 0 longitude")
