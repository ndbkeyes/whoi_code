# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 10:36:25 2021

@author: ndbke
"""

import xarray as xr
import numpy as np
import haversine as hv
import matplotlib.pyplot as plt

file_t = 'C:/Users/ndbke/Dropbox/_NDBK/Research/WHOI/WOA_data/woa18_A5B7_t00_04.nc'
file_s = 'C:/Users/ndbke/Dropbox/_NDBK/Research/WHOI/WOA_data/woa18_A5B7_s00_04.nc'

# open NetCDF files for temperature and salinity WOA data
nc_t = xr.open_dataset(file_t, decode_times=False)
nc_s = xr.open_dataset(file_s, decode_times=False)

# print(nc_t)


lon = nc_t.coords["lon"]
lat = nc_t.coords["lat"]
depth = nc_t.coords["depth"]
lon_size = nc_t.sizes["lon"]
lat_size = nc_t.sizes["lat"]
depth_size = nc_t.sizes["depth"]



vol = xr.DataArray( np.zeros((lon_size,lat_size,depth_size)) , coords=[lon,lat,depth], dims=["lat","lon","depth"])

# I = lon.values + 1/8
# II = lon.values - 1/8
# III = lat.values + 1/8
# IV = lat.values - 1/8
# V = lon.values

# AB = hv.haversine( (I[0],IV[0]) , (II[0],IV[0]) )
# CD = hv.haversine( (I[0],III[0]), (II[0],III[0]) )
# EF = hv.haversine( (V[0],IV[0]),  (V[0],III[0]) )

# area = 1/2 * (AB+CD) * EF
# print(area)


# for i in range(0,lon_size-1):
#     for j in range(0,lat_size-1):
#         for k in range(0,depth_size-1):
    
area = xr.DataArray( np.zeros((lat_size,lon_size)) , coords=[lat.values,lon.values], dims=["lat","lon"])
    

for i in range(0,lat_size-1):
    for j in range(0,lon_size-1):
        
            lat_val = lat.values[i]
            lon_val = lon.values[j]
            
            # haversine is in km!
            vert = hv.haversine( (lat_val - 1/8, lon_val), (lat_val + 1/8, lon_val) )
            hori = hv.haversine( (lat_val, lon_val - 1/8), (lat_val, lon_val + 1/8) )
            
            area.values[i,j] = vert * hori

            

area.plot()

print(np.amax(area.values))
