# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 16:55:09 2021

@author: ndbke
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize



#%% UNIVARIATE constraint to solve for Lagrange multiplier
def f_1v(beta,X,Xavg):
    
    num = 0
    denom = 0
    for i in range(len(X)):
        # print("f_1v:",i,beta,X[i])
        num += X[i] * beta**X[i]
        denom += beta**X[i]
        
    return num / denom - Xavg




#%% BIVARIATE constraint to solve for Lagrange multipliers
def f_2v(bg,X,Xavg,Y,Yavg):
    
    beta, gamma = bg.T
    
    num_X = 0
    denom_X = 0
    num_Y = 0
    denom_Y = 0
    
    for i in range(len(X)):
        for j in range(len(Y)):
            
            # T constraint values
            num_X += X[i] * beta**X[i] * gamma**Y[j]
            denom_X += beta**X[i] * gamma**Y[j]
            
            # S constraint values
            num_Y += Y[j] * beta**X[i] * gamma**Y[j]
            denom_Y += beta**X[i] * gamma**Y[j]        
            
    
    X_val = num_X / denom_X - Xavg
    Y_val = num_Y / denom_Y - Yavg
    
    return np.array([X_val, Y_val])




# UNIVARIATE entropy maximization
def max_1v(X,Xavg,max_guess=20):
    
    # adapt range of guesses to the inputted value
    guesses = np.linspace(0.001,max_guess,2000)
    # explore for roots across guess space
    roots = optimize.newton(f_1v, x0=guesses, args=(X,Xavg),maxiter=1000,full_output=True)
    # look at roots for which the method converged 
    legit_roots = roots.root[roots.converged]
    # eliminate nans
    legit_roots = legit_roots[~np.isnan(legit_roots)]
    # round roots so we don't double count
    round_to = int(np.floor(np.log(max_guess)*1.5))
    legit_roots = np.round(legit_roots, round_to)
    # unique root values
    legit_roots = np.unique(legit_roots)
    
    # get beta and alpha from root
    beta = -np.log(legit_roots)
    alpha = np.log( np.sum( np.exp(-beta * X) ) )
    
    p_arr_max = np.empty((len(X),1))
    for i in range(0,len(X)):
        p_arr_max[i] = np.exp( -alpha - beta * X[i] )
    p_arr_max /= np.sum(p_arr_max)
    
    return legit_roots, beta, p_arr_max
    
    
    
    




#%% scripting

# test-inputs
X = np.array([-1.25,-0.75,-0.25])
Xavg = -0.96
Y = np.linspace(29,35,10)
Yavg = 31
beta = np.array([2.5,2.75,3])
gamma = np.array([1.5,1.75,2])
bg = np.column_stack((beta,gamma))


# 1D maximization
xroot, xbeta, xpam = max_1v(X,Xavg)
yroot, ybeta, ypam = max_1v(Y,Yavg)
fx = f_1v(xroot,X,Xavg)
fy = f_1v(yroot,Y,Yavg)


# plot 1D constraints & roots
maxx = np.max([xroot,yroot])*1.1
xplot = np.linspace(0.001,maxx,100)
fplotx = f_1v(xplot,X,Xavg)
fploty = f_1v(xplot,Y,Yavg)
plt.figure()
plt.hlines(0,0,maxx,colors='green')
plt.plot(xplot,fplotx)
plt.scatter(xroot,fx)
plt.plot(xplot,fploty)
plt.scatter(yroot,fy)
plt.legend(["x","y"])


# plot pmaxarr solution
plt.figure()
plt.plot(X,xpam)
plt.figure()
plt.plot(Y,ypam)

# xxplot = np.column_stack((xplot,xplot))
# fplot2, fplot3 = f_2v(xxplot, X,Xavg,Y,Yavg)