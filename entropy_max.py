# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 13:06:20 2021

@author: ndbke
"""



from scipy import optimize
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt



def f(y,T,Tavg):
     
    T1 = T[0]
    Tarr = T[1:]
    return (Tavg - T1) + np.sum( (Tavg - Tarr) * y**(Tarr - T1) )


def fp1(y,T,Tavg):
    
    T1 = T[0]
    Tarr = T[1:]    
    return np.sum( (Tavg - Tarr) * (Tarr - T1) * y**(Tarr - T1 - 1) )


def fp2(y,T,Tavg):
    
    T1 = T[0]
    Tarr = T[1:]    
    return np.sum( (Tavg - Tarr) * (Tarr - T1) * (Tarr - T1 - 1) * y**(Tarr - T1 - 2) )
        


def max_ent(T,Tavg):
   
    y = optimize.newton(f, x0=1, fprime=fp1, fprime2=fp2, args=(T,Tavg))
    beta = -np.log(y)
    alpha = np.log( np.sum( np.exp(-beta * T) ) )
    
    print(f"beta = {np.round(beta,4)}, alpha = {np.round(alpha,4)}")
    
    p_arr = np.empty((len(T),1))
    for i in range(0,len(T)):
        p_arr[i] = np.exp( -alpha - beta * T[i] )
    p_arr /= np.sum(p_arr)
        
    print(np.round(p_arr,4))
    
    
    
#%%    

# file_data = 'NetCDFs/tsv_arc.nc'
# dat_arc = xr.open_dataset(file_data, decode_times=False, autoclose=True)

# T = dat_arc.t
# print(T)

    
T = np.array([-1.25, -0.75, -0.25])
Tavg = -0.96
max_ent(T,Tavg)


