# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 16:55:09 2021

@author: ndbke
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize


### 1-variable constraint for Lagrange multiplier
def f_1v(beta,X,Xavg):
    
    num = 0
    denom = 0
    for i in range(len(X)):
        num += X[i] * beta**X[i]
        denom += beta**X[i]
        
    return num / denom - Xavg




### 2-variable constraint for Lagrange multipliers
def f_2v(bg,X,Xavg,Y,Yavg):
    
    beta, gamma = bg
    
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
    
    return X_val, Y_val




### 1-variable entropy maximization
def max_1v(X,Xavg):
    
    # adapt range of guesses to the inputted value
    max_guess = 50
    guesses = np.linspace(0.001,max_guess,200)
    # explore for roots across guess space
    roots = optimize.newton(f_1v, x0=guesses, args=(X,Xavg),maxiter=1000,full_output=True)
    
    roots = clean_roots(roots)
    
    if roots.size == 0:
        print('no roots')
        return
    
    # get beta and alpha from root
    beta = -np.log(roots[0])
    alpha = np.log( np.sum( np.exp(-beta * X) ) )
    
    # get 
    p_arr_max = np.empty((len(X),1))
    for i in range(0,len(X)):
        p_arr_max[i] = np.exp( -alpha - beta * X[i] )
    p_arr_max /= np.sum(p_arr_max)
    
    return roots, beta, p_arr_max




### 2-variable entropy maximization
def max_2v(X,Xavg,Y,Yavg):
    
    # adapt range of guesses to the inputted value
    guesses = np.column_stack((np.linspace(0.001,50,200),np.linspace(0.001,50,200))).T
    # explore for roots across guess space
    roots = optimize.newton(f_2v, x0=guesses, args=(X,Xavg,Y,Yavg),maxiter=1000,full_output=True)
    # clean up returned roots
    roots = clean_roots(roots)
    
    # define Lagrange multipliers
    beta = -np.log(roots[0])
    gamma = -np.log(roots[1])
    alpha = np.log( np.sum( np.exp(-beta * X) ) * np.sum( np.exp(-gamma * Y) ) )

    # get maxconent distribution
    p_arr = np.empty((len(Y),len(X)))
    for i in range(0,len(Y)):
        for j in range(0,len(X)):
            p_arr[i,j] = np.exp( -alpha - beta * X[j] - gamma * Y[i] )
    p_arr /= np.sum(p_arr)

    return roots, [alpha,beta,gamma], p_arr



### find actual roots output by Newton's Method
def clean_roots(roots_obj):
    
    # only roots for which alg converged
    roots = roots_obj.root[roots_obj.converged] 
    
    # remove NaNs
    roots = roots[~np.isnan(roots)] 

    # round off to remove tiny differences between what are actually same roots         
    roots = np.round(roots,5)                   
    
    # get unique values
    roots = np.unique(roots)   
                 
    return roots
    
    



#%% test-inputs
X = np.linspace(-2,12,10)
Xavg = 3
Y = np.linspace(32,35,10)
Yavg = 34





#%% UNIVARIATE TEST


# 1D maximization
xroot, xbeta, xpam = max_1v(X,Xavg)
yroot, ybeta, ypam = max_1v(Y,Yavg)
fx = f_1v(xroot,X,Xavg)
fy = f_1v(yroot,Y,Yavg)
print(xroot, yroot)

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


# plot 1D solutions
plt.figure()
plt.plot(X,xpam)
plt.figure()
plt.plot(Y,ypam)




#%% BIVARIATE TEST

# 2D maximization
roots, mults, p_arr = max_2v(X,Xavg,Y,Yavg)

# plot 2D solution
plt.figure()
plt.pcolormesh(Y,X,p_arr.T,shading="nearest")
plt.colorbar()
