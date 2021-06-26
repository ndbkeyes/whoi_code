# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 09:32:36 2021

@author: ndbke
@
"""


import numpy as np
import xarray as xr
import gsw
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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
mask1 = (dat.lat > 60) & (dat.lat < 65) & (dat.lon > 150)  & (dat.lon < 180)
# leftmost part of Pacific - near Alaska
mask2 = (dat.lat > 60) & (dat.lat < 65) & (dat.lon > -180) & (dat.lon < -160)
# areas by Nordic countries
mask3 = (dat.lat > 60) & (dat.lat < 67) & (dat.lon > 17)   & (dat.lon < 40)
# Canadian island water & Hudson Bay
mask4 = (dat.lat > 60) & (dat.lat < 72) & (dat.lon > -130) & (dat.lon < -70)


# set colors for mask plotting
ncm1 = ListedColormap(np.array([0.5,0.5,0.5]))
ncm2 = ListedColormap(np.array([0,0,0]))

# plot mask over SA data
mask_plot = mask1 | mask2 | mask3 | mask4
plt.figure()
dat.CT.isel(depth=0).plot()
mask_plot = mask_plot.where(mask_plot == 1)
mask_plot.plot(alpha=0.05,cmap=ncm1)
plt.ylim(60,90)


plt.figure()
globe = dat.CT.isel(depth=0).plot(robust=True,subplot_kws=dict(projection=ccrs.Orthographic(-20, 90), facecolor="white"),transform=ccrs.PlateCarree())
mask_plot = mask_plot.where(mask_plot == 1)
mask_plot.plot(alpha=0.025,ax=globe.axes,cmap=ncm1,transform=ccrs.PlateCarree())
globe.axes.set_global()
globe.axes.coastlines()
globe.axes.set_extent([20,200,20,200])



# filter out data in masking areas
cond = (~mask1) & (~mask2) & (~mask3) & (~mask4)
cond = (~mask1) & (~mask2) & (~mask3)
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

# res1: 0.1, 0.05
# res2: 0.5, 0.25
# res3: 1, 0.5
t_increment = 0.1
s_increment = 0.05
T_bins = np.arange(-3,13,t_increment)
S_bins = np.arange(22,37,s_increment)

# bin each T, S value
T_dig = np.digitize(T,T_bins)
S_dig = np.digitize(S,S_bins)

# 2D matrix to hold volumes
V_matrix = np.zeros((len(T_bins)-1,len(S_bins)-1))



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




#%% PLOTTING


print("plotting")
fig, axes = plt.subplots()




### isopycnals

# make empty array with coordinates of CT and SA
pot_dens = xr.DataArray(np.zeros((len(S_bins),len(T_bins))), coords=[S_bins,T_bins], dims=["SA","CT"] )
# fill matrix with GSW potential density values
pot_dens.values = gsw.sigma0( pot_dens.SA, pot_dens.CT )
# transpose to get CT on vertical axis
pot_dens = pot_dens.transpose()
# plot isopycnals!
pdi = pot_dens.plot.contour(ax=axes, colors='blue',linewidths=0.4,levels=12)
axes.clabel(pdi, pdi.levels, fontsize=6)


### freezing line

# make empty array with coordinates of SA only
freeze_pt = xr.DataArray( np.zeros((len(S_bins))), coords=[S_bins], dims=["SA"] )
# fill array with GSW freezing point values
freeze_pt.values = gsw.CT_freezing(freeze_pt.SA, 0, 0)
# plot freezing line
fpl = freeze_pt.plot(ax=axes, color="green",linestyle="dashed",linewidth=1)




###  volumetric T-S

pcm = axes.pcolormesh(S_bins,T_bins,V_matrix,cmap="YlOrRd", norm=colors.LogNorm())
cbar = plt.colorbar(pcm)
cbar.set_label("volume (km$^3$)")
plt.xlabel("absolute salinity (g/kg)")
plt.ylabel("conservative temperature (Celsius)")
plt.title("Volumetric T-S plot for Arctic Ocean")

plt.xlim(np.amin(S_bins),np.amax(S_bins))
plt.ylim(np.amin(T_bins),np.amax(T_bins))

# plt.xlim(25,35)
# plt.ylim(-2,1.5)

plt.savefig("../plots/tsv_arc.png",dpi=200)






#%% save TSV to NetCDF

print("saving")

tsv = xr.DataArray( V_matrix , coords=[T_bins,S_bins], dims=["T","S"])
tsv.name = "volume"
tsv.to_netcdf("NetCDFs/tsv_arc.nc")
tsv.close()
