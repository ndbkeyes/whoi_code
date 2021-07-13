# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 11:12:54 2021

@author: ndbke
"""


import numpy as np
from scipy import optimize
import xarray as xr


# univariate / marginal Shannon information entropy
def H1(p_arr):
    
    H = 0
    for p in p_arr:
        if p != 0:
            H += p * np.log2(p)
    H *= -1
    
    if isinstance(p_arr,xr.core.dataarray.DataArray):
        H = np.round(H.values[0],3)
    else:
        H = np.round(H,3)
    
    return H
    
    
    
# bivariate / joint Shannon information entropy
def H2(p_grid):
    
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
    
    if isinstance(p_grid,xr.core.dataarray.DataArray):
        H = np.round(H.values[0],4)
    else:
        H = np.round(H,4)
    
    return H
    

# conditional Shannon information entropy
def Hc(p_arr,p_grid):
    
    H = H2(p_grid) - H1(p_arr)
    
    return np.round(H,4)




# joint Shannon information entropy
def J(p1_arr,p2_arr,p_grid):
    
    J = H1(p1_arr) + H1(p2_arr) - H2(p_grid)
    return np.round(J,3)



# entropy quantities of vol T-S xarray p_TS
def entropy_all(p_TS,disp=False):
    
    # set NaN values to zero
    p_TS.values = np.nan_to_num(p_TS.values)

    # get totals by T and S classes separately
    tdim = p_TS.dims[0]
    sdim = p_TS.dims[1]
    p_T = p_TS.sum(sdim)
    p_S = p_TS.sum(tdim)
    
    # normalize all to get frequencies (probabilities)
    V_total = p_TS.sum()
    p_TS = p_TS / V_total
    p_T = p_T / V_total
    p_S = p_S / V_total
    
    # print results if desired
    if disp:
        print("H(T) - marginal entropy:\t\t",H1(p_T))
        print("H(S) - marginal entropy:\t\t",H1(p_S))
        print("H(T)/H(S) - entropy ratio:\t\t",np.round(H1(p_T)/H1(p_S),2))
        print("H(T,S) - joint entropy:\t\t\t",H2(p_TS))
        print("H_S(T) - conditional entropy:\t",Hc(p_S,p_TS))
        print("H_T(S) - conditional entropy:\t",Hc(p_T,p_TS))
        print("J(T,S) - dependence metric:\t\t", J(p_T,p_S,p_TS),"\n")
        
    return H1(p_T), H1(p_S), H2(p_TS), J(p_T,p_S,p_TS)






# maximization constraint polynomial to solve for Lagrange mults & max prob dist
def f(y,X,Xavg):
    X1 = X[0]
    Xarr = X[1:]
    return (Xavg - X1) + np.sum( (Xavg - Xarr) * y**(Xarr - X1) )

# max constraint polynomial's derivative
def fp1(y,X,Xavg):
    X1 = X[0]
    Xarr = X[1:]    
    return np.sum( (Xavg - Xarr) * (Xarr - X1) * y**(Xarr - X1 - 1) )

# max constraint polynomial's second derivative
def fp2(y,X,Xavg):
    X1 = X[0]
    Xarr = X[1:]    
    return np.sum( (Xavg - Xarr) * (Xarr - X1) * (Xarr - X1 - 1) * y**(Xarr - X1 - 2) )
        


# maximum possible entropy for n states (uniform probability!)
def Hmax(n):
    return np.round(np.log2(n),4)



# univariate constrained entropy maximization
def max_ent1(X,Xavg):
   
    # solve polynomial equation
    root = optimize.newton(f, x0=1, fprime=fp1, fprime2=fp2, args=(X,Xavg))
    
    # beta, alpha from root
    beta = -np.log(root)
    alpha = np.log( np.sum( np.exp(-beta * X) ) )
    
    # max-entropy probability distribution
    p_arr = np.empty((len(X),1))
    for i in range(0,len(X)):
        p_arr[i] = np.exp( -alpha - beta * X[i] )
    p_arr /= np.sum(p_arr)
    
    # maximum constrainted entropy value 
    Hp_max = H1(p_arr)
    H_max = Hmax(len(X))
    
    # print & return values
    print(f"beta = {np.round(beta,4)}, alpha = {np.round(alpha,4)}")
    print(np.round(p_arr,4))
    print(Hp_max)
    print(H_max)
    return alpha, beta, p_arr, Hp_max, H_max




    