# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 14:09:01 2021

@author: ndbke
"""

import xarray as xr
import numpy as np
import gsw



def prep_data(latlon_bounds=[-90,90,-180,180],name="all"):
    

    
    # load files for T and S
    file_t = 'C:/Users/ndbke/Dropbox/_NDBK/Research/WHOI/data/woa18_A5B7_t00_04.nc'
    file_s = 'C:/Users/ndbke/Dropbox/_NDBK/Research/WHOI/data/woa18_A5B7_s00_04.nc'
    nc_t = xr.open_dataset(file_t, decode_times=False, autoclose=True)
    nc_s = xr.open_dataset(file_s, decode_times=False, autoclose=True)
    
    # get correct DataArrays
    dat_t = nc_t.t_an
    dat_s = nc_s.s_an
    
    

    
    # filter data to Arctic Ocean at t=0
    cond = (dat_t.lat >= latlon_bounds[0]) & (dat_t.lat <= latlon_bounds[1]) & (dat_t.lon >= latlon_bounds[2]) & (dat_t.lon <= latlon_bounds[3])
    
    dat_t = dat_t.isel(time=0)
    dat_t = dat_t.where( cond )
    
    dat_s = dat_s.isel(time=0)
    dat_s = dat_s.where( cond )
    
    # print resulting arrays
    print(dat_t)
    print(dat_s)
    
    

    # make dataset with in-situ T, S
    dataset = xr.Dataset(
        {
            "temperature": dat_t,
            "salinity": dat_s
        },
        coords={
            "lat": dat_t.coords["lat"],
            "lon": dat_t.coords["lon"],
            "depth": dat_t.coords["depth"]
        }
    )
    
    
    # add conservative temperature & absolute salinity & assoc quantities to dataset
    dataset['pressure'] = xr.apply_ufunc( gsw.p_from_z, -dataset.depth, np.mean(dataset.lat) )
    dataset['SA']       = xr.apply_ufunc( gsw.SA_from_SP, dataset.salinity, dataset.pressure, dataset.lon, dataset.lat )
    dataset['pot_temp'] = xr.apply_ufunc( gsw.pt0_from_t, dataset.SA, dataset.temperature, dataset.pressure )
    dataset['CT']       = xr.apply_ufunc( gsw.CT_from_pt, dataset.SA, dataset.pot_temp )
    dataset['pot_dens'] = xr.apply_ufunc( gsw.sigma0, dataset.SA, dataset.CT )
    

    # print & save to file
    print(dataset)
    fn = "../NetCDFs/data_{label}.nc"
    dataset.to_netcdf(fn.format(label=name),mode='w')
    
    
    
    print("done")
    
    
    
    
prep_data()
