# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 17:51:55 2021

@author: ndbke
"""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

ax = plt.axes(projection = ccrs.NorthPolarStereo())
ax.coastlines()
ax.set_title("Arctic Ocean")
ge = ax.get_extent()
ax.set_extent([-5000000,5000000,-5000000,5000000],crs=ccrs.NorthPolarStereo())
plt.show()
