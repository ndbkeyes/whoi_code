# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 10:56:00 2021

@author: ndbke
"""


import matplotlib.pyplot as plt
import matplotlib.colors as colors


# wrapper function on pcolormesh to adapt to TSV xarray
def plot_tsv(xarr,xylabels=["absolute salinity (g/kg)","conservative temperature (C)"],corner="UR"):
    
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
    
        
    pcm = plt.pcolormesh(xarr.salinity,xarr.temperature,pltmat,norm=colors.LogNorm(),cmap="YlOrRd")
    cbar = plt.colorbar(pcm)
    cbar.set_label("volume (km$^3$)")    
    
    plt.xlabel(xylabels[0])
    plt.ylabel(xylabels[1])