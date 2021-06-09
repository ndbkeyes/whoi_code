# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 10:36:25 2021

@author: ndbke
"""

import xarray as xr
import numpy as np
import haversine as hv
import matplotlib.pyplot as plt





#%%#############################
### get and filter T, S data ###
################################


# filenames for T and S data files
file_t = 'C:/Users/ndbke/Dropbox/_NDBK/Research/WHOI/WOA_data/woa18_A5B7_t00_04.nc'
file_s = 'C:/Users/ndbke/Dropbox/_NDBK/Research/WHOI/WOA_data/woa18_A5B7_s00_04.nc'

# get Datasets of WOA temperature and salinity from NetCDF files
nc_t = xr.open_dataset(file_t, decode_times=False)
nc_s = xr.open_dataset(file_s, decode_times=False)

# get correct T, S data arrays out
data_t = nc_t["t_an"]
data_s = nc_s["s_an"]

# filter them to just include Arctic ocean below 200m, at the one time we have
data_t = data_t.isel(time=0)
data_t = data_t.where(data_t.depth >= 200.0)
data_t = data_t.where(data_t.lat > 60.0)
data_s = data_s.isel(time=0)
data_s = data_s.where(data_s.depth >= 200.0)
data_s = data_s.where(data_s.lat > 60.0)

# get coordinate values - same for T and S, I checked
lon = nc_t.coords["lon"].values
lat = nc_t.coords["lat"].values
depth = nc_t.coords["depth"].values

# get lengths of coordinate arrays
lon_size = nc_t.sizes["lon"]
lat_size = nc_t.sizes["lat"]
depth_size = nc_t.sizes["depth"]




#%%########################################
### find volume around each observation ###
###########################################


# make empty DataArrays for plotting quantities  
area = xr.DataArray( np.empty((lat_size,lon_size)) , coords=[lat,lon], dims=["lat","lon"])
crossec = xr.DataArray( np.empty((depth_size,lat_size)) , coords=[depth,lat], dims=["depth","lat"])
area.values[:] = np.nan
crossec.values[:] = np.nan

# make empty DataArray for volumes
vol = xr.DataArray( np.empty((depth_size,lat_size,lon_size)) , coords=[depth,lat,lon], dims=["depth","lat","lon"])
vol.values[:] = np.nan


# make array for potential temperature
pT = data_t.values


### LAT ###
for i in range(0,lat_size-1):
    # skip if lat is not in Arctic
    if lat[i] < 60:
        continue
    print("lat: ",lat[i])
    
    ### LON ###
    for j in range(0,lon_size-1):
        
        # find distances N/S, E/W around point, 1/8" on each side
        length = hv.haversine( (lat[i] - 1/8, lon[j]), (lat[i] + 1/8, lon[j]) )
        width = hv.haversine( (lat[i], lon[j] - 1/8), (lat[i], lon[j] + 1/8) )
        # find & store approx. surface area around current point
        area.values[i,j] = length * width
            
        ### DEPTH ###
        for k in range(0,depth_size):
            
            # skip if not deep enough
            if depth[k] < 200:
                continue
            # top of water column
            if k == 0:
                height = 1/2 * ( depth[k+1] - depth[k] )
            # bottom of water column
            elif k == depth_size-1:
                height = 1/2 * ( depth[k] - depth[k-1] )  
            # interior
            else:
                height = 1/2 * ( depth[k+1] - depth[k-1] )
            # factor of 1/1000 to convert from m to km !!!
            height *= 1/1000
            # find & store approx. volume around current point in DataArray
            # index order is k,i,j because we want to match the dimension order of T, S DataArrays
            vol.values[k,i,j] = length * width * height
            # store volume in cross-section also
            crossec.values[k,i] = length * width * height
            
            
            
            
            
            
print("done")  



#%%##################################################
### plot areas and longitudinal cross-sec volumes ###
#####################################################       

plt.figure()
area.plot()
plt.title("Sector area by (lat,lon)")

plt.figure() 
crossec.plot()
plt.gca().invert_yaxis()
plt.title("Volume by (lat,depth)")




#%%#################################################
### convert to pot temp & save as Dataset/NetCDF ###
####################################################


dataset = xr.Dataset(
    {
        "pot_temp": (["depth","lat","lon"],pT),
        "salinity": (["depth","lat","lon"],data_s.values),
        "volume": (["depth","lat","lon"],vol.values),
    },
    coords={
        "lat": data_t.coords["lat"],
        "lon": data_t.coords["lon"],
        "depth": data_t.coords["depth"],
    }, 
)
print(dataset)

dataset.to_netcdf("tsv.nc")

