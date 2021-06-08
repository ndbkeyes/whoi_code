# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 09:32:36 2021

@author: ndbke
@
"""


import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

#%%


# open Dataset with T, S, V - so don't have to find V every time!
file_tsv = 'C:/Users/ndbke/Dropbox/_NDBK/Research/WHOI/whoi_code/tsv.nc'
nc_tsv = xr.open_dataset(file_tsv, decode_times=False)
print(nc_tsv)

# get T, S, V value arrays out
temperature = nc_tsv["temperature"].values
salinity = nc_tsv["salinity"].values
volume = nc_tsv["volume"].values




#%%

### HISTOGRAM AND PLOT T, S FROM MY NETCDF DATSET

## hist and geog plots of T
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


## hist and geog plots of S
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


#%%

# flatten all three arrays into 1D
T = temperature.flatten()
S = salinity.flatten()
V = volume.flatten()

# take out NaNs
nan_bool = ~np.isnan(T) & ~np.isnan(S)
T = T[nan_bool]
S = S[nan_bool]
V = V[nan_bool]

N_points = len(T)

t_increment = 0.25
s_increment = 0.025

# make bin edges for T-S diagram
T_bins = np.arange(-2.5,10,t_increment)
S_bins = np.arange(32.5,35.5,s_increment)


# get bin indices for each datapoint in T and S
T_dig = np.digitize(T,T_bins)
S_dig = np.digitize(S,S_bins)


# get bin-center points
T_points = T_bins[0:len(T_bins)-2] + t_increment/2
S_points = S_bins[0:len(S_bins)-2] + s_increment/2


# make empty 2D matrix to hold volumes
V_matrix = np.zeros((len(T_bins),len(S_bins)))


for i in range(0,N_points):
    t_ind = T_dig[i]-1
    s_ind = S_dig[i]-1
    # subtracted 1 to convert bin number (starting at 1) to array index (starting at 0)
    
    if ~np.isnan(V[i]):
        V_matrix[len(T_bins)-t_ind,s_ind] += V[i]


V_matrix[V_matrix == 0] = np.nan

plt.figure()
Vflat = V_matrix.flatten()
plt.hist(Vflat,bins=100)

plt.figure()
plt.pcolormesh(S_bins,T_bins,V_matrix)
plt.colorbar()
