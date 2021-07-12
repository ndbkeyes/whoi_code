# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 14:07:56 2021

@author: ndbke
"""

import xarray as xr
import matplotlib.pyplot as plt
from utils.make_tsv import make_tsv
from utils.entropy import entropy_all
from utils.plot_tsv import plot_tsv


def geog_rect(xarr,lims=0):
    
    if lims == 0:
        print("error - no bounds given")
        return 0
    
    if len(lims['lat']) == 0:
        lims['lat'] = [-90,90]
    if len(lims['lon']) == 0:
        lims['lon'] = [-180,180]
    
    cond = (xarr.lat >= lims['lat'][0]) & (xarr.lat <= lims['lat'][1]) & (xarr.lon >= lims['lon'][0]) & (xarr.lon <= lims['lon'][1])
    sel = xarr.where(cond)
    
    return sel
 

print("opening & filtering files")

#%% WOA TSV - GRN

# open data
file_data = 'NetCDFs/data_all.nc'
dat = xr.open_dataset(file_data, decode_times=False, autoclose=True, mode="a")
dat.close()

# open bathymetry mask
file_bath = 'NetCDFs/bathymetry.nc'
bath = xr.open_dataset(file_bath, decode_times=False, autoclose=True)
bath.close()

# open volume matrix
file_vol = 'NetCDFs/volume.nc'
vol = xr.open_dataset(file_vol, decode_times=False, autoclose=True)
vol.close()

# filter data by location and bathymetry
dat = geog_rect(dat, {'lat': [70,80], 'lon': [-20,15]})
dat = dat.where(bath.bath_mask)



#%%

# make vol T-S from WOA
tsv_grn = make_tsv(dat,vol,res=[0.5,0.05],tsbounds=[-2,8,32,35.3],name="grn",convert=False)



#%% select different water masses

# make and apply condition on T, S coords
## deepwater has slightly different placement than in CMK- look into this?
cond = (tsv_grn.t >= -1.5) & (tsv_grn.t < 0) & (tsv_grn.s >= 34.8) & (tsv_grn.s < 34.9)
deep = tsv_grn.where(cond)
upper = tsv_grn.where(~cond)

# plot upper and lower water masses on T-S
plt.figure()
plot_tsv(upper,xylabels=["salinity","potential temperature"])
plt.figure()
plot_tsv(deep,xylabels=["salinity","potential temperature"])


#%% entropy on Greenland TSV
print("ALL - GRN")
entropy_all(tsv_grn,disp=True)

print("DEEP GRN")
entropy_all(deep,disp=True)
print("UPPER GRN")
entropy_all(upper,disp=True)
