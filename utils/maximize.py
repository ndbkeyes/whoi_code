# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 11:26:33 2021

@author: ndbke
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from utils.entropy import H1, H2
import matplotlib.colors as colors



#%% polynomials for maximization

### UNIVARIATE (vectorized)

# vectorized maximum entropy constraint polynomial
def f_vec(y,X,Xavg):
    
    X1 = X[0]
    Xarr = X[1:]
    
    ans = np.array([])
    for i in range(len(y)):
        val = (Xavg - X1) + np.sum( (Xavg - Xarr) * y[i]**(Xarr - X1) )
        ans = np.append(ans,val)
        
    return ans




# vectorized first-deriv polynomial
def fp1_vec(y,X,Xavg):
    
    X1 = X[0]
    Xarr = X[1:]
    
    ans = np.array([])
    for i in range(len(y)):
        val = np.sum( (Xavg - Xarr) * (Xarr - X1) * y[i]**(Xarr - X1 - 1) )
        ans = np.append(ans,val)
        
    return ans



# vectorized second-deriv polynomial
def fp2_vec(y,X,Xavg):
    
    X1 = X[0]
    Xarr = X[1:]
    
    ans = np.array([])
    for i in range(len(y)):
        val = np.sum( (Xavg - Xarr) * (Xarr - X1) * (Xarr - X1 - 1) * y[i]**(Xarr - X1 - 2) )
        ans = np.append(ans,val)
        
    return ans





### UNIVARIATE (unvectorized)

# max constraint to solve
def f(y,X,Xavg):
    X1 = X[0]
    Xarr = X[1:]
    return (Xavg - X1) + np.sum( (Xavg - Xarr) * y**(Xarr - X1) )

# max constraint's 1st deriv
def fp1(y,X,Xavg):
    X1 = X[0]
    Xarr = X[1:]    
    return np.sum( (Xavg - Xarr) * (Xarr - X1) * y**(Xarr - X1 - 1) )

# max constraint's 2nd deriv
def fp2(y,X,Xavg):
    X1 = X[0]
    Xarr = X[1:]    
    return np.sum( (Xavg - Xarr) * (Xarr - X1) * (Xarr - X1 - 1) * y**(Xarr - X1 - 2) )
        




### BIVARIATE (unvectorized)

# 2vbl constraint to solve (fixed errors!)
def f_TS(xy,X,Xavg,Y,Yavg):
    
    x, y = xy
    
    num_X = 0
    denom_X = 0
    num_Y = 0
    denom_Y = 0
    
    for i in range(len(X)):
        for j in range(len(Y)):
            
            # T constraint values
            num_X += X[i] * x**X[i] * y**Y[j]
            denom_X += x**X[i] * y**Y[j]
            
            # S constraint values
            num_Y += Y[j] * x**X[i] * y**Y[j]
            denom_Y += x**X[i] * y**Y[j]        
            
    
    X_val = num_X / denom_X - Xavg
    Y_val = num_Y / denom_Y - Yavg
    
    return np.array([X_val, Y_val])



#%%







def poly_solve(f,fp1,fp2,arg_tuple,plot=False,guesses=0):
    
    X = arg_tuple[0]
    Xavg = arg_tuple[1]
    
    # default array of guesses
    if guesses == 0:
        guesses = np.linspace(0,200,1000)
    
    # run Newton's Method
    root = optimize.newton(f, x0=guesses, fprime=fp1, fprime2=fp2, args=arg_tuple, maxiter=10000, full_output=True)

    
    # get the roots that CONVERGED
    actual_roots = root.root[root.converged]

    # round all of them off so that slightly different ones become the same
    actual_roots = np.unique(np.round(actual_roots,4))
    
    # eliminate false roots (that don't actualy have a value close to zero)
    actual_vals = f(actual_roots,X,Xavg)
    # do it by checking against the smallest value we have - sorta assumes taht at least one root is real but w/e idk
    actual_roots = actual_roots[abs(actual_vals) <= abs(2 * np.nanmin(actual_vals))]
    
    
    print("roots:", actual_roots)
    
    # plot function and roots
    if plot:
        plt.figure()
        plt.plot(guesses,f(guesses,X,Xavg))
        plt.xlim(0,20)
        plt.ylim(-1000,1000)
        plt.scatter(actual_roots,f(actual_roots,X,Xavg))
    
    return actual_roots



#%% one and two variable constrained entropy maximization



# univariate maximization
def max_ent1(X,Xavg):
   
    # solve polynomial equation
    root = optimize.newton(f, x0=0.1, fprime=fp1, fprime2=fp2, args=(X,Xavg), maxiter=1000)
    # root = poly_solve(f,fp1,fp2,(X,Xavg),plot=True)
    
    # beta, alpha from root
    beta = -np.log(root)
    alpha = np.log( np.sum( np.exp(-beta * X) ) )
    
    # max-entropy probability distribution
    p_arr = np.empty((len(X),1))
    for i in range(0,len(X)):
        p_arr[i] = np.exp( -alpha - beta * X[i] )
    p_arr /= np.sum(p_arr)
    
    # plt.figure()
    # plt.plot(X,p_arr)
    
    # maximum constrainted entropy value 
    Hp_max = H1(p_arr)
    H_max = np.round(np.log2(len(X)),4)
    
    # print & return values
    print(f"beta = {np.round(beta,4)}, alpha = {np.round(alpha,4)}")
    print(np.round(p_arr,4))
    print(Hp_max)
    print(H_max)
    
    # plt.plot(X,p_arr)
    # plt.vlines(Xavg,0,1)
    
    return p_arr



# # bivariate maximization
# # WRONG, used incorrect math for the constraints
# def max_ent2(X,Xavg, Y,Yavg):
    
#     # solve polynomial equations
#     root_x = optimize.newton(f, x0=10, fprime=fp1, fprime2=fp2, args=(X,Xavg))
#     root_y = optimize.newton(f, x0=10, fprime=fp1, fprime2=fp2, args=(Y,Yavg))
#     print("found roots")
    
#     # beta, gamma, alpha from roots
#     beta = -np.log(root_x)
#     gamma = -np.log(root_y)
#     alpha = np.log( np.sum( np.exp(-beta * X) ) * np.sum( np.exp(-gamma * Y) ) )
    
#     # max-entropy probability distribution
#     p_arr = np.empty((len(Y),len(X)))
#     for i in range(0,len(Y)):
#         for j in range(0,len(X)):
#             p_arr[i,j] = np.exp( -alpha - beta * X[j] - gamma * Y[i] )
#     p_arr /= np.sum(p_arr)
    
#     # maximum constrainted entropy value 
#     Hp_max = H2(p_arr)
    
#     # print & return values
#     print(f"beta = {np.round(beta,4)}, gamma = {np.round(gamma,4)}, alpha = {np.round(alpha,4)}")
#     print(np.round(p_arr,4))
#     print(np.round(Hp_max,4))
#     # return alpha, beta, p_arr, Hp_max, H_max
    
#     plt.figure()
#     plt.pcolormesh(X,Y,p_arr,shading="nearest",cmap="YlOrRd")
#     plt.colorbar()
    
#     return p_arr


# BIVARIATE FIXED
def max_ent2(X,Xavg, Y,Yavg):
    
    g = np.array([0.1,0.1])
    [root_x, root_y] = optimize.newton(f_TS, g, args=(X,Xavg, Y,Yavg))
    
    
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
    print("probability array:", np.round(p_arr,4))
    print("Hpmax:", np.round(Hp_max,4))
    # return alpha, beta, p_arr, Hp_max, H_max
    
    plt.figure()
    pcm = plt.pcolormesh(X,Y,p_arr,shading='nearest',cmap="YlOrRd")
    pcm.norm = colors.LogNorm()
    plt.colorbar()
    plt.xlabel("salinity (o/oo)")
    plt.ylabel("potential temperature (C)")
    
#     return p_arr
    
    return root_x, root_y


#%% finding distributions of actual TSV space


def tsv_dists(xarr):
    
    # np.log(xarr).plot()
    
    p_T = xarr.sum('s')
    p_S = xarr.sum('t')
    p_T /= np.sum(p_T)
    p_S /= np.sum(p_S)
    
    T_avg = np.average(xarr.t.values,weights=p_T)
    S_avg = np.average(xarr.s.values,weights=p_S)

    print("T and S averages:", np.round(T_avg,3), np.round(S_avg,3))
    
    plt.figure()
    plt.plot(xarr.t.values,p_T.values)
    
    plt.figure()
    plt.plot(xarr.s.values,p_S.values)
    
    return T_avg, S_avg




def tsv_dists2(xarr):
    
    # np.log(xarr).plot()
    
    p_T = xarr.sum('s')
    p_S = xarr.sum('t')
    p_T /= np.sum(p_T)
    p_S /= np.sum(p_S)
    
    return p_T, p_S
 
    
#%%


