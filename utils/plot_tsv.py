# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 10:56:00 2021

@author: ndbke
"""


import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


# wrapper function on pcolormesh to adapt to TSV xarray
def plot_tsv(xarr,xylabels=["s","t"],corner="BR",norm="log"):
    
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
    
    # 2D colorplot with colorbar    
    pcm = plt.pcolormesh(xarr.s,xarr.t,pltmat,cmap="YlOrRd")
    if norm == "log":
        pcm.norm = colors.LogNorm()
        plt.clim(1,10**6)
    elif norm == "log_signed":
        pcm.norm = colors.LogNorm()
        plt.clim(10**-4,10**4)
    cbar = plt.colorbar(pcm)
    cbar.set_label("volume (km$^3$)")    
    
    # labels and limits
    plt.xlabel(xylabels[0])
    plt.ylabel(xylabels[1])
    plt.xlim(np.amin(xarr.coords["s"]),np.amax(xarr.coords["s"]))
    plt.ylim(np.amin(xarr.coords["t"]),np.amax(xarr.coords["t"]))
    
    