# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 17:21:26 2021

@author: ndbke
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors




def plot_tsv(xarr,xylabels,corner="UR"):
    
    # assuming that coordinates are bins' left/lower edges
    
    pltmat = xarr.values
    
    # trim xarray's values matrix
    if corner == "UL":
        pltmat = pltmat[1:,1:]
    elif corner == "UR":
        pltmat = pltmat[1:,0:-1]
    elif corner == "BR":
        pltmat = pltmat[0:-1,0:-1]
    elif corner == "BL":
        pltmat = pltmat[0:-1,1:]
    
        
    pcm = plt.pcolormesh(xarr.salinity,xarr.temperature,pltmat,norm=colors.LogNorm())
    cbar = plt.colorbar(pcm)
    cbar.set_label("volume (km$^3$)")    
    
    plt.xlabel(xylabels[0])
    plt.ylabel(xylabels[1])
    
    
    



# read in CSV file
grn_tsv = np.genfromtxt("../carmack_tsv.csv", delimiter=',')

# get T and S coordinates
T_bins = grn_tsv[1:,0]
S_bins = grn_tsv[0,1:]

print(T_bins)
print(S_bins)

# trim T, S coordinates out of data matrix
grn_tsv = grn_tsv[1:,1:]


# plot volumetric T-S
# can't use xr.plot() because it doesn't let you do binning correctly
grn_trim = grn_tsv[1:,0:-1]     # reduced dimensionality to plot w/ bin edges
pcm = plt.pcolormesh(S_bins,T_bins,grn_trim,norm=colors.LogNorm())
cbar = plt.colorbar(pcm)
cbar.set_label("volume (km$^3$)")
plt.xlabel("salinity (o/oo)")
plt.ylabel("potential temperature (Celsius)")
plt.title("Carmack & Aagard - Volumetric T-S plot for Greenland Sea")
plt.xlim(33.75,np.amax(S_bins))
plt.ylim(np.amin(T_bins),np.amax(T_bins))


# save as DataArray & NetCDF
tsv = xr.DataArray( grn_tsv , coords=[T_bins,S_bins], dims=["temperature","salinity"])
tsv.name = "volume"
tsv.to_netcdf("NetCDFs/tsv_grn.nc")
tsv.close()





#%% select different water masses

# make and apply condition on coords
cond = (tsv.temperature >= -1.5) & (tsv.temperature < 0) & (tsv.salinity >= 34.85) & (tsv.salinity < 34.95)
deep = tsv.where(cond)
upper = tsv.where(~cond)

plt.figure()
plot_tsv(upper,["salinity","potential temperature"])
plt.figure()
plot_tsv(deep,["salinity","potential temperature"])




