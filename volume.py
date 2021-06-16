# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 10:36:25 2021

@author: ndbke
"""

import xarray as xr
import numpy as np
import haversine as hv
import matplotlib.pyplot as plt





#%% get prepared data from NetCDF file!
file_t = 'C:/Users/ndbke/Dropbox/_NDBK/Research/WHOI/whoi_code/NetCDFs/data.nc'
dat = xr.open_dataset(file_t, decode_times=False, autoclose=True)
print(dat)


# get coordinate values
lon = dat.coords["lon"].values
lat = dat.coords["lat"].values
depth = dat.coords["depth"].values


#%% find volume around each observation


# make empty DataArrays for plotting quantities  
area    = xr.DataArray( np.empty( (len(lat),   len(lon)) ) , coords=[lat,lon],   dims=["lat","lon"])
crossec = xr.DataArray( np.empty( (len(depth), len(lat)) ) , coords=[depth,lat], dims=["depth","lat"])
area.values[:] = np.nan
crossec.values[:] = np.nan


# make empty DataArray for volumes
vol      = xr.DataArray( np.empty((len(depth),len(lat),len(lon))) , coords=[depth,lat,lon], dims=["depth","lat","lon"])
vol.name = "volume"
vol.values[:] = np.nan



### LAT ###
for i in range(0,len(lat)):
    
    # mask to Arctic Ocean
    if lat[i] < 60:
        continue

    print("lat: ",lat[i])
    
    ### LON ###
    for j in range(0,len(lon)):
        
        # find distances N/S, E/W around point, 1/8" on each side
        length = hv.haversine( (lat[i] - 1/8, lon[j]), (lat[i] + 1/8, lon[j]) )
        width = hv.haversine( (lat[i], lon[j] - 1/8), (lat[i], lon[j] + 1/8) )
        
        # find & store approx. surface area around current point
        area.values[i,j] = length * width
            
        ### DEPTH ###
        for k in range(0,len(depth)):
            
            # top of water column
            if k == 0:
                height = 1/2 * ( depth[k+1] - depth[k] )
            # bottom of water column
            elif k == len(depth)-1:
                height = 1/2 * ( depth[k] - depth[k-1] )  
            # interior
            else:
                height = 1/2 * ( depth[k+1] - depth[k-1] )
                
            # factor of 1/1000 to convert from m to km
            height *= 1/1000
            
            # find & store approx. volume around current point in DataArray
            # index order is k,i,j because we want to match the dimension order of T, S DataArrays
            vol.values[k,i,j] = length * width * height
            # store volume in cross-section also
            crossec.values[k,i] = length * width * height




#%% save volume as NetCDF
vol.to_netcdf("NetCDFs/volume.nc")


#%% plot areas and longitudinal cross-sec volumes 

plt.figure()
area.plot()
plt.title("Sector area by (lat,lon)")

plt.figure() 
crossec.plot()
plt.gca().invert_yaxis()
plt.title("Volume by (lat,depth)")

