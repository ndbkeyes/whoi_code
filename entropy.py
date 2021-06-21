# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 16:37:23 2021

@author: ndbke

"""

import numpy as np

def entropy1(p_arr):
    
    H = 0
    for p in p_arr:
        H += p * np.log2(p)
    H *= -1
    
    print(H)
    
    
    
def entropy2(p_grid):
    
    
    dim = p_grid.shape
    H = 0
    for i in dim[0]:
        for j in dim[1]:
            p = p_grid[i,j]
            H += p * np.log2(p)
    H *= -1
    
    print(H)


N_upper = 210543
N_deep = 1230957
N = N_upper + N_deep

p_upper = N_upper / N
p_deep = N_deep / N

print(p_upper,p_deep)


