# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 09:32:36 2021

@author: ndbke
@
"""


import numpy as np
import xarray as xr
import gsw
import matplotlib.pyplot as plt

from utils.plot_tsv import plot_tsv




def make_tsv(dat_TS,vol,res=[0.5,0.25],tsbounds=[-2,8,32,36],name="",convert=True):
    
    #%% create V matrix by T, S
    
    print("binning T, S")
    
    # flatten all three arrays into 1D
    if convert:
        T = dat_TS.CT.values.flatten()
        S = dat_TS.SA.values.flatten()
    else:
        T = dat_TS.temperature.values.flatten()
        S = dat_TS.salinity.values.flatten()
    
    V = vol.volume.values.flatten()
    
    # # plot data at 200m depth
    # plt.figure()
    # dat_TS.CT.isel(depth=24).plot()

    # remove NaNs from arrays
    nan_bool = ~np.isnan(T) & ~np.isnan(S)
    T = T[nan_bool]
    S = S[nan_bool]
    V = V[nan_bool]
    
    # make T-S bins using inputted params
    T_bins = np.arange(tsbounds[0],tsbounds[1]+res[0],res[0])
    S_bins = np.arange(tsbounds[2],tsbounds[3]+res[1],res[1])
    
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
    
    # make TSV into a DataArray
    tsv = xr.DataArray( V_matrix, coords=[T_bins,S_bins], dims=["t","s"])
    tsv.name = "volume"
    #*** note that for some reason we can't use uppercase T as a dim - may already have some xarray use???
    
    
    #%% PLOTTING
    print("plotting")
    fig, axes = plt.subplots()
    
    
    
    ### isopycnals
    # empty array with coordinates of CT and SA
    pot_dens = xr.DataArray(np.zeros((len(S_bins),len(T_bins))), coords=[S_bins,T_bins], dims=["SA","CT"] )
    # fill matrix with GSW potential density values
    pot_dens.values = gsw.sigma0( pot_dens.SA, pot_dens.CT )
    # transpose to get CT on vertical axis
    pot_dens = pot_dens.transpose()
    # plot isopycnals!
    pdi = pot_dens.plot.contour(ax=axes, colors='blue',linewidths=0.4,levels=12)
    axes.clabel(pdi, pdi.levels, fontsize=6)
    
    
    ### freezing line
    # empty array with coordinates of SA only
    freeze_pt = xr.DataArray( np.zeros((len(S_bins))), coords=[S_bins], dims=["SA"] )
    # fill array with GSW freezing point values
    freeze_pt.values = gsw.CT_freezing(freeze_pt.SA, 0, 0)
    # plot freezing line
    freeze_pt.plot(ax=axes, color="green",linestyle="dashed",linewidth=1)
    
    
    ###  volumetric T-S
    plot_tsv(tsv,corner="BR")
    
    
    #%% save TSV to NetCDF
    
    print("saving")
    plt.savefig(f"../plots/tsv_{'name'}.png",dpi=200)
    tsv.to_netcdf(f"NetCDFs/tsv_{'name'}.nc")
    tsv.close()
    
    
    return tsv
