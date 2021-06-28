# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 12:00:45 2021

@author: ndbke
"""


import xarray as xr
from utils.make_tsv import make_tsv



#%% open NetCDF files

print("opening & filtering files")
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


# filter out data in masking areas
cond = (~mask1) & (~mask2) & (~mask3) & (~mask4)
dat["CT"] = dat.CT.where( cond )
dat["SA"] = dat.SA.where( cond )

dat.close()
vol.close()


# make TSV out of the filtered data
make_tsv(dat,vol,res=[0.1,0.05],name="arc")
