# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 16:37:23 2021

@author: ndbke

"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt



def entropy1(p_arr):
    
    H = 0
    
    for p in p_arr:
        if p != 0:
            H += p * np.log2(p)
            
    H *= -1
    H = np.round(H.values,2)
    
    return H
    
    
    
def entropy2(p_grid):
    
    
    dim = p_grid.shape
    T_dim = dim[0]
    S_dim = dim[1]
    H = 0
    
    for i in range(T_dim):
        for j in range(S_dim):
            p = p_grid[i,j]
            if p != 0:
                H += p * np.log2(p)
    
    H *= -1
    H = np.round(H.values,2)
    
    return H
    

def entropyc(p_arr,p_grid):
    
    H = entropy2(p_grid) - entropy1(p_arr)
    
    return np.round(H,2)
    


#%%


file_tsv = 'NetCDFs/tsv.nc'
dat = xr.open_dataset(file_tsv, decode_times=False, autoclose=True)

p_TS = dat.volume
p_TS.values = np.nan_to_num(p_TS.values)

p_T = p_TS.sum("S")
p_S = p_TS.sum("T")

V_total = p_TS.sum()
p_TS = p_TS / V_total
p_T = p_T / V_total
p_S = p_S / V_total


print("T marginal entropy:",entropy1(p_T))
print("S marginal entropy:",entropy1(p_S))

print("T-S joint entropy:",entropy2(p_TS))

print("T conditional entropy:",entropyc(p_S,p_TS))
print("S conditional entropy:",entropyc(p_T,p_TS))
