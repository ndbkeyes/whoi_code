# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 17:21:26 2021

@author: ndbke
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors



# read in CSV file
grn_tsv = np.genfromtxt("../carmack_tsv.csv", delimiter=',')

# get T and S coordinates
T_bins = grn_tsv[1:,0]
S_bins = grn_tsv[0,1:]

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
tsv = xr.DataArray( grn_tsv , coords=[T_bins,S_bins], dims=["T","S"])
tsv.name = "grn_volume"
tsv.to_netcdf("NetCDFs/carmack_tsv.nc")
tsv.close()
