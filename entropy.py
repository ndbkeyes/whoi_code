# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 16:37:23 2021

@author: ndbke

"""

import numpy as np
import xarray as xr


# univariate / marginal Shannon information entropy
def shanent1(p_arr):
    
    H = 0
    
    for p in p_arr:
        if p != 0:
            H += p * np.log2(p)
            
    H *= -1
    H = np.round(H.values,2)
    
    return H
    
    
    
# bivariate / joint Shannon information entropy
def shanent2(p_grid):
    
    
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
    

# conditional Shannon information entropy
def shanentc(p_arr,p_grid):
    
    H = shanent2(p_grid) - shanent1(p_arr)
    
    return np.round(H,2)





def J(p1_arr,p2_arr,p_grid):
    
    J = shanent1(p1_arr) + shanent1(p2_arr) - shanent2(p_grid)
    return np.round(J,2)




#%%


# read in volumetric T-S
file_tsv = 'NetCDFs/tsv_grn.nc'
dat = xr.open_dataset(file_tsv, decode_times=False, autoclose=True)
dat.close()


print("Entropy quantities - Greenland Sea")

# get volumes by T-S class, set NaN's to zero
p_TS = dat.volume
p_TS.values = np.nan_to_num(p_TS.values)

# get totals by T and S classes separately
p_T = p_TS.sum("S")
p_S = p_TS.sum("T")

# normalize all to get frequencies (probabilities)
V_total = p_TS.sum()
p_TS = p_TS / V_total
p_T = p_T / V_total
p_S = p_S / V_total


print("H(T) - marginal entropy:\t\t",shanent1(p_T))
print("H(S) - marginal entropy:\t\t",shanent1(p_S))
print("H(T)/H(S) - entropy ratio:\t\t",np.round(shanent1(p_T)/shanent1(p_S),2))
print("H(T,S) - joint entropy:\t\t\t",shanent2(p_TS))
print("H_S(T) - conditional entropy:\t",shanentc(p_S,p_TS))
print("H_T(S) - conditional entropy:\t",shanentc(p_T,p_TS))
print("J(T,S) - dependence metric:\t\t", J(p_T,p_S,p_TS))




