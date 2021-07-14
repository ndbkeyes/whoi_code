# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 11:26:33 2021

@author: ndbke
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from utils.entropy import H1, H2



#%% polynomials for maximization


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
        



#%% one and two variable constrained entropy maximization



# univariate maximization
def max_ent1(X,Xavg):
   
    # solve polynomial equation
    root = optimize.newton(f, x0=10, fprime=fp1, fprime2=fp2, args=(X,Xavg))
    
    # beta, alpha from root
    beta = -np.log(root)
    alpha = np.log( np.sum( np.exp(-beta * X) ) )
    
    # max-entropy probability distribution
    p_arr = np.empty((len(X),1))
    for i in range(0,len(X)):
        p_arr[i] = np.exp( -alpha - beta * X[i] )
    p_arr /= np.sum(p_arr)
    
    plt.figure()
    plt.plot(X,p_arr)
    
    # maximum constrainted entropy value 
    Hp_max = H1(p_arr)
    H_max = np.round(np.log2(len(X)),4)
    
    # print & return values
    print(f"beta = {np.round(beta,4)}, alpha = {np.round(alpha,4)}")
    print(np.round(p_arr,4))
    print(Hp_max)
    print(H_max)
    return alpha, beta, p_arr, Hp_max, H_max



# bivariate maximization
def max_ent2(X,Xavg, Y,Yavg):
    
    # solve polynomial equations
    root_x = optimize.newton(f, x0=10, fprime=fp1, fprime2=fp2, args=(X,Xavg))
    root_y = optimize.newton(f, x0=10, fprime=fp1, fprime2=fp2, args=(Y,Yavg))
    print("found roots")
    
    # beta, gamma, alpha from roots
    beta = -np.log(root_x)
    gamma = -np.log(root_y)
    alpha = np.log( np.sum( np.exp(-beta * X) ) * np.sum( np.exp(-gamma * Y) ) )
    
    # max-entropy probability distribution
    p_arr = np.empty((len(Y),len(X)))
    for i in range(0,len(Y)):
        for j in range(0,len(X)):
            p_arr[i,j] = np.exp( -alpha - beta * X[j] - gamma * Y[i] )
    p_arr /= np.sum(p_arr)
    
    # maximum constrainted entropy value 
    Hp_max = H2(p_arr)
    
    # print & return values
    print(f"beta = {np.round(beta,4)}, gamma = {np.round(gamma,4)}, alpha = {np.round(alpha,4)}")
    print(np.round(p_arr,4))
    print(np.round(Hp_max,4))
    # return alpha, beta, p_arr, Hp_max, H_max
    
    plt.figure()
    plt.pcolormesh(X,Y,p_arr,shading="nearest",cmap="YlOrRd")
    plt.colorbar()
    
    return p_arr


#%% finding distributions of actual TSV space


def tsv_dists(xarr):
    
    p_T = xarr.sum('s')
    p_S = xarr.sum('t')
    p_T /= np.sum(p_T)
    p_S /= np.sum(p_S)
    
    T_avg = np.average(xarr.t.values,weights=p_T)
    S_avg = np.average(xarr.s.values,weights=p_S)

    print(T_avg, S_avg)
    
    plt.figure()
    plt.plot(xarr.t.values,p_T.values)
    
    plt.figure()
    plt.plot(xarr.s.values,p_S.values)
    
    return T_avg, S_avg
 