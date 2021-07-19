# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 16:37:23 2021

@author: ndbke

"""


import xarray as xr
import numpy as np
from utils.entropy import entropy_all
from utils.maximize import *
import matplotlib.pyplot as plt


#%%


# read in volumetric T-S files
tsv_grn = xr.open_dataset('NetCDFs/tsv_grn.nc', decode_times=False, autoclose=True)
tsv_arc = xr.open_dataset('NetCDFs/tsv_arc.nc', decode_times=False, autoclose=True)
tsv_cmk = xr.open_dataset('NetCDFs/tsv_cmk.nc', decode_times=False, autoclose=True)

# entropy calcs for entirety of Greenland Sea and Arctic Ocean
print("grn")
entropy_all(tsv_grn.volume,True)
print("cmk")
entropy_all(tsv_cmk.volume,True)

# condition to get Greenland deep water vs. upper water
cond_grn = (tsv_grn.t >= -1.5) & (tsv_grn.t < 0) & (tsv_grn.s >= 34.8) & (tsv_grn.s <= 34.95)
deep_grn = tsv_grn.where(cond_grn)
upper_grn = tsv_grn.where(~cond_grn)

# entropy calcs for Greenland deep vs upper
print("grn - deep")
entropy_all(deep_grn.volume,True)
print("grn - upper")
entropy_all(upper_grn.volume,True)


cond_cmk = (tsv_cmk.t >= -1.5) & (tsv_cmk.t < 0) & (tsv_cmk.s >= 34.8) & (tsv_cmk.s <= 34.95)
deep_cmk = tsv_cmk.where(cond_cmk)
upper_cmk = tsv_cmk.where(~cond_cmk)

# entropy calcs for CMK deep vs upper
print("cmk - deep")
entropy_all(deep_cmk.volume,True)
print("cmk - upper")
entropy_all(upper_cmk.volume,True)

# close NC files
tsv_grn.close()
tsv_arc.close()
tsv_cmk.close()





#%% univariate testing

T_test = np.array([-1.25,-0.75,-0.25])
Tavg_test = -0.96

plt.figure()
p_arr = max_ent1(T,Tavg_test)
plt.plot(T_test,p_arr)
plt.xlabel("T")
plt.ylabel("probability")

p_T, p_S = tsv_dists2(deep_cmk.volume)
plt.plot(T_test,p_T[1:4])
plt.legend(["maximized distribution","actual distribution"])

print(H1(p_T).values)


cut_dcv = deep_cmk.volume[1:4,-10:-8]
print(H2(cut_dcv / np.sum(cut_dcv)))


#%% bivariate testing

from utils.maximize import *

[Tavg, Savg] = tsv_dists(deep_cmk.volume)
print(Tavg,Savg)

root_x, root_y = max_ent2(np.array([34.85,34.95]),34.94, np.array([-1.25,-0.75,-0.25]),-0.96)
print("roots:",np.round(root_x,3), np.round(root_y,3))

plt.figure()
plt.xlabel("x & y")
plt.ylabel("f")
plt.legend(["f(x)","f(y)"])
plt.scatter(root_x,f_TS([root_x,root_y],T,Tavg,S,Savg)[0])
plt.scatter(root_y,f_TS([root_x,root_y],T,Tavg,S,Savg)[1])