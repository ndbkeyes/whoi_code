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
from matplotlib.colors import ListedColormap


#%% open NetCDF files

print("opening & filtering files")

# file_data = 'NetCDFs/data.nc'
# dat_orig = xr.open_dataset(file_data, decode_times=False, autoclose=True)

# open bathymetrically masked data
file_bath = 'NetCDFs/data_isobath.nc'
dat = xr.open_dataset(file_bath, decode_times=False, autoclose=True)

# open volume matrix
file_vol = 'NetCDFs/volume.nc'
vol = xr.open_dataset(file_vol, decode_times=False, autoclose=True)




# rightmost part of Pacific - near Russia
mask1 = (dat.lat > 60) & (dat.lat < 70) & (dat.lon > 150)  & (dat.lon < 180)
# areas by Nordic countries
mask2 = (dat.lat > 60) & (dat.lat < 67) & (dat.lon > 17)   & (dat.lon < 40)
# leftmost part of Pacific - near Alaska
mask3 = (dat.lat > 60) & (dat.lat < 65) & (dat.lon > -180) & (dat.lon < -170)
# Canadian island water & Hudson Bay
mask4 = (dat.lat > 60) & (dat.lat < 72) & (dat.lon > -130) & (dat.lon < -70)


# set colors for mask plotting
ncm1 = ListedColormap(np.array([0.5,0,0.9]))
ncm2 = ListedColormap(np.array([0,0,0]))

# plot mask over SA data
mask_plot = mask1 | mask2 | mask3 | mask4
plt.figure()
dat.SA.isel(depth=0).plot()
mask_plot = mask_plot.where(mask_plot == 1)
mask_plot.plot(alpha=0.05,cmap=ncm1)
plt.ylim(60,90)

# filter out data in masking areas
cond = (~mask1) & (~mask2) & (~mask3) & (~mask4)
dat["CT"] = dat.CT.where( cond )
dat["SA"] = dat.SA.where( cond )



#%% create V matrix by T, S

print("binning T, S")


# flatten all three arrays into 1D
T = dat.CT.values.flatten()
S = dat.SA.values.flatten()
V = vol.volume.values.flatten()

dat.close()
vol.close()

# remove NaNs
nan_bool = ~np.isnan(T) & ~np.isnan(S)
T = T[nan_bool]
S = S[nan_bool]
V = V[nan_bool]

# make T-S bins
t_increment = 0.5
s_increment = 0.25
T_bins = np.arange(-3,13,t_increment)
S_bins = np.arange(22,37,s_increment)

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




#%% add isopycnals and freezing line

print("plotting")
fig, axes = plt.subplots()


# make empty array with coordinates of CT and SA
pot_dens = xr.DataArray(np.zeros((len(S_bins),len(T_bins))), coords=[S_bins,T_bins], dims=["SA","CT"] )
# fill matrix with GSW potential density values
pot_dens.values = gsw.sigma0( pot_dens.SA, pot_dens.CT )
# transpose to get CT on vertical axis
pot_dens = pot_dens.transpose()
# plot isopycnals!
pdi = pot_dens.plot.contour(ax=axes, colors='blue',linewidths=0.4,levels=12)
axes.clabel(pdi, pdi.levels, fontsize=6)

# make empty array with coordinates of SA only
freeze_pt = xr.DataArray( np.zeros((len(S_bins))), coords=[S_bins], dims=["SA"] )
# fill array with GSW freezing point values
freeze_pt.values = gsw.CT_freezing(freeze_pt.SA, 0, 0)
# plot freezing line
fpl = freeze_pt.plot(ax=axes, color="green",linestyle="dashed",linewidth=1)




#%% plot volumetric T-S

pcm = axes.pcolormesh(S_bins,T_bins,V_matrix,cmap="YlOrRd", norm=colors.LogNorm())
cbar = plt.colorbar(pcm)
cbar.set_label("volume (km$^3$)")
plt.xlabel("absolute salinity (g/kg)")
plt.ylabel("conservative temperature (Celsius)")
plt.title("Volumetric T-S plot for Arctic Ocean")

plt.xlim(np.amin(S_bins),np.amax(S_bins))
plt.ylim(np.amin(T_bins),np.amax(T_bins))

plt.savefig("../plots/tsv.png",dpi=200)

