# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 09:32:36 2021

@author: ndbke
@
"""


import numpy as np
import xarray as xr
import matplotlib.pyplot as plt



#%%#####################
### open TSV dataset ###
########################


# open Dataset
file_tsv = 'C:/Users/ndbke/Dropbox/_NDBK/Research/WHOI/whoi_code/tsv.nc'
nc_tsv = xr.open_dataset(file_tsv, decode_times=False)
print(nc_tsv)

# get value arrays out
temperature = nc_tsv["temperature"].values
salinity = nc_tsv["salinity"].values
volume = nc_tsv["volume"].values

depth = nc_tsv.coords["depth"].values
lat = nc_tsv.coords["lat"].values
lon = nc_tsv.coords["lon"].values



#%%##########################
### histogram & plot T, S ###
#############################

t = nc_tsv["temperature"]
t = t.isel(depth=24)
tv = t.values
plt.figure()
plt.hist(tv.flatten(),bins=50)
plt.xlim(-2,10)
print(t)
plt.figure()
t.plot(robust=True)
plt.ylim(60,90)

s = nc_tsv["salinity"]
s = s.isel(depth=24)
sv = s.values
plt.figure()
plt.hist(sv.flatten(),bins=100)
plt.xlim(33,36)
print(s)
plt.figure()
s.plot(robust=True)
plt.ylim(60,90)

# close NetCDF filestream
nc_tsv.close()


#%%############################
### create V matrix by T, S ###
###############################



# flatten all three arrays into 1D
T = temperature.flatten()
S = salinity.flatten()
V = volume.flatten()

# take out NaNs (points w/ no observations)
nan_bool = ~np.isnan(T) & ~np.isnan(S)
T = T[nan_bool]
S = S[nan_bool]
V = V[nan_bool]

# get number of points left to analyze
N_points = len(T)




# make bin edges for T-S diagram
t_increment = 0.125
s_increment = 0.0125
T_bins = np.arange(-3,11,t_increment)
S_bins = np.arange(32,36,s_increment)

# get bin indices for each datapoint in T and S
T_dig = np.digitize(T,T_bins)
S_dig = np.digitize(S,S_bins)

# make empty 2D matrix to hold volumes
V_matrix = np.zeros((len(T_bins),len(S_bins)))




# add up volumes for each T-S state
for i in range(0,N_points):
    t_ind = T_dig[i]-1
    s_ind = S_dig[i]-1
    # subtracted 1 to convert bin number (starting at 1) to array index (starting at 0)
    
    if ~np.isnan(V[i]):
        V_matrix[t_ind,s_ind] += V[i]
        
        


# get rid of any T-S states with zero volume
V_matrix[V_matrix == 0] = np.nan

# log-transform to make the plot more readable
V_matrix[:] = np.log(V_matrix[:])

# histogram of volumes
plt.figure()
V_flat = V_matrix.flatten()
plt.hist(V_flat,bins=100)





#%%#############################
### make volumetric T-S plot ###
################################


# 2D color plot of volumetric T-S
plt.figure()
plt.pcolormesh(S_bins,T_bins,V_matrix,cmap="YlOrRd")
cbar = plt.colorbar()
cbar.set_label("Volume (km^3)")
plt.xlabel("salinity")
plt.ylabel("temperature")
plt.title("Volumetric T-S plot for Arctic Ocean")
