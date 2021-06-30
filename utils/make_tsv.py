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




def make_tsv(dat_TS,vol,res=[0.5,0.25],name=""):
    
    #%% create V matrix by T, S
    
    print("binning T, S")
    
    
    # flatten all three arrays into 1D
    T = dat_TS.CT.values.flatten()
    S = dat_TS.SA.values.flatten()
    V = vol.volume.values.flatten()
    
    plt.figure()
    dat_TS.CT.isel(depth=0).plot()

    # remove NaNs from arrays
    nan_bool = ~np.isnan(T) & ~np.isnan(S)
    T = T[nan_bool]
    S = S[nan_bool]
    V = V[nan_bool]
    
    # make T-S bins
    
    # res1: 0.1, 0.05
    # res2: 0.5, 0.25
    # res3: 1, 0.5
    t_increment = res[0]
    s_increment = res[1]
    T_bins = np.arange(-2,8,t_increment)
    S_bins = np.arange(32,36,s_increment)
    
    print(T_bins)
    
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
    tsv = xr.DataArray( V_matrix, coords=[T_bins,S_bins], dims=["temperature","salinity"])
    
    
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
    freeze_pt.plot(ax=axes, color="green",linestyle="dashed",linewidth=1)
    
    
    ###  volumetric T-S
    plot_tsv(tsv,corner="BR")
    # plt.xlim(np.amin(S_bins),np.amax(S_bins))
    # plt.ylim(np.amin(T_bins),np.amax(T_bins))
    
    
    
    
    
    
    
    #%% save TSV to NetCDF
    
    print("saving")
    plt.savefig(f"../plots/tsv_{'name'}.png",dpi=200)
    tsv = xr.DataArray( V_matrix , coords=[T_bins,S_bins], dims=["T","S"])
    tsv.name = "volume"
    tsv.to_netcdf("NetCDFs/tsv_arc.nc")
    tsv.close()
