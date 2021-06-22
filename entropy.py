# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 16:37:23 2021

@author: ndbke

"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def shanent1(p_arr):
    
    H = 0
    
    for p in p_arr:
        if p != 0:
            H += p * np.log2(p)
            
    H *= -1
    H = np.round(H.values,2)
    
    return H
    
    
    
def shanent2(p_grid):
    
    
    dim = p_grid.shape
    T_dim = dim[0]
    S_dim = dim[1]
    H = 0
    
    for i in range(T_dim):
        for j in range(S_dim):
            p = p_grid[i,j]
            if p != 0:
                H += p * np.log2(p)
    
    H *= -1
    H = np.round(H.values,2)
    
    return H
    

def shanentc(p_arr,p_grid):
    
    H = shanent2(p_grid) - shanent1(p_arr)
    
    return np.round(H,2)
    


#%%


file_tsv = 'NetCDFs/tsv.nc'
dat = xr.open_dataset(file_tsv, decode_times=False, autoclose=True)

p_TS = dat.volume
p_TS.values = np.nan_to_num(p_TS.values)

p_T = p_TS.sum("S")
p_S = p_TS.sum("T")

V_total = p_TS.sum()
p_TS = p_TS / V_total
p_T = p_T / V_total
p_S = p_S / V_total


print("T marginal entropy:",shanent1(p_T))
print("S marginal entropy:",shanent1(p_S))

print("T-S joint entropy:",shanent2(p_TS))

print("T conditional entropy:",shanentc(p_S,p_TS))
print("S conditional entropy:",shanentc(p_T,p_TS))

dat.close()



#%%

# read in CSV file
grn_tsv = np.genfromtxt("../carmack_tsv.csv", delimiter=',')

# get T and S coordinates
T_bins = grn_tsv[1:,0]
S_bins = grn_tsv[0,1:]

# trim T, S coordinates out of data matrix
grn_tsv = grn_tsv[1:,1:]


# plot volumetric T-S
# can't use xr.plot() because it doesn't let you do binning correctly
grn_trim = grn_tsv[1:,0:-1]     # needed to reduce dimensionality to plot with bin edges
pcm = plt.pcolormesh(S_bins,T_bins,grn_trim,norm=colors.LogNorm())
cbar = plt.colorbar(pcm)
cbar.set_label("volume (km$^3$)")
plt.xlabel("salinity (o/oo)")
plt.ylabel("potential temperature (Celsius)")
plt.title("Carmack & Aagard - Volumetric T-S plot for Greenland Sea")
plt.xlim(33.75,np.amax(S_bins))
plt.ylim(np.amin(T_bins),np.amax(T_bins))


# save as DataArray & NetCDF
tsv = xr.DataArray( grn_tsv , coords=[T_bins,S_bins], dims=["T","S"])
tsv.name = "grn_volume"
tsv.to_netcdf("NetCDFs/carmack_tsv.nc")
tsv.close()


