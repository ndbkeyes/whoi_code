# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 09:32:36 2021

@author: ndbke
@
"""


import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors


#%% open NetCDF files

print("opening & filtering files")

# open bathymetrically masked data
file_bath = 'NetCDFs/data_bath.nc'
dat = xr.open_dataset(file_bath, decode_times=False, autoclose=True)

# open volume matrix
file_vol = 'NetCDFs/volume.nc'
vol = xr.open_dataset(file_vol, decode_times=False, autoclose=True)

mask1 = (dat.lat > 60) & (dat.lat < 70) & (dat.lon > 150)  & (dat.lon < 180)
mask2 = (dat.lat > 60) & (dat.lat < 67) & (dat.lon > 17)   & (dat.lon < 40)
mask3 = (dat.lat > 60) & (dat.lat < 70) & (dat.lon > -180)  & (dat.lon < -170)
mask4 = (dat.lat > 60) & (dat.lat < 73) & (dat.lon > -130) & (dat.lon < -70)
dat["CT"] = dat.CT.where( (~mask1) & (~mask2) & (~mask3) & (~mask4) )
dat["SA"] = dat.SA.where( (~mask1) & (~mask2) & (~mask3) & (~mask4) )

# plt.figure()
# mask1.plot()

# plt.figure()
# mask2.plot()

plt.figure()
dat.SA.isel(depth=60).plot(robust=True)

#%% create V matrix by T, S

print("binning T, S")


# flatten all three arrays into 1D
T = dat.CT.values.flatten()
S = dat.SA.values.flatten()
V = vol.volume.values.flatten()

# remove NaNs
nan_bool = ~np.isnan(T) & ~np.isnan(S)
T = T[nan_bool]
S = S[nan_bool]
V = V[nan_bool]

# make T-S bins
t_increment = 0.1
s_increment = 0.05
T_bins = np.arange(-3,12,t_increment)
S_bins = np.arange(23,36.5,s_increment)

# bin each T, S value
T_dig = np.digitize(T,T_bins)
S_dig = np.digitize(S,S_bins)

# 2D matrix to hold volumes
V_matrix = np.zeros((len(T_bins),len(S_bins)))



#%% fill V matrix

print("filling V matrix")

### add up volumes for each T-S state
for i in range(0,len(T)):
    
    # -1 converts bin number to array index
    t_ind = T_dig[i]-1
    s_ind = S_dig[i]-1
    
    # add volume to T-S matrix cell
    if ~np.isnan(V[i]):
        V_matrix[t_ind,s_ind] += V[i]
        

# get rid of any T-S states with zero volume
V_matrix[V_matrix == 0] = np.nan


#%% make volumetric T-S plot

print("plotting")

plt.figure()
plt.pcolormesh(S_bins,T_bins,V_matrix,cmap="YlOrRd", norm=colors.LogNorm())
cbar = plt.colorbar()
cbar.set_label("volume (km$^3$)")
plt.xlabel("absolute salinity (g/kg)")
plt.ylabel("conservative temperature (Celsius)")
plt.title("Volumetric T-S plot for Arctic Ocean")



#%%
dat.close()
vol.close()

print("done")




