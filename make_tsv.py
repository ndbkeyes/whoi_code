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
import gsw


#%% open TSV dataset

# open Dataset
file_tsv = 'C:/Users/ndbke/Dropbox/_NDBK/Research/WHOI/whoi_code/tsv.nc'
dat = xr.open_dataset(file_tsv, decode_times=False)
print(dat)



#%% convert to conservative temperature (CT) and absolute salinity (SA)

dat['pressure'] = xr.apply_ufunc( gsw.p_from_z, -dat.depth, dat.lat )
dat['SA']       = xr.apply_ufunc( gsw.SA_from_SP, dat.salinity, dat.pressure, dat.lon, dat.lat )
dat['pot_temp'] = xr.apply_ufunc( gsw.pt0_from_t, dat.SA, dat.temperature, dat.pressure )
dat['CT']       = xr.apply_ufunc( gsw.CT_from_pt, dat.SA, dat.pot_temp )
dat['pot_dens'] = xr.apply_ufunc( gsw.sigma0, dat.SA, dat.CT )

dat.close()


#%% create V matrix by T, S



# flatten all three arrays into 1D
T = dat.CT.values.flatten()
S = dat.SA.values.flatten()
V = dat.volume.values.flatten()

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


#%% make volumetric T-S plot

plt.figure()
plt.pcolormesh(S_bins,T_bins,V_matrix,cmap="YlOrRd", norm=colors.LogNorm())
cbar = plt.colorbar()
cbar.set_label("volume (km^3)")
plt.xlabel("salinity (g/kg)")
plt.ylabel("potential temperature (Celsius)")
plt.title("Volumetric T-S plot for Arctic Ocean")
