import xarray as xr
import matplotlib.pyplot as plt
from haversine import haversine


#%%#############################
### open and read data files ###
################################

file_t = 'C:/Users/ndbke/Dropbox/_NDBK/Research/WHOI/WOA_data/woa18_A5B7_t00_04.nc'
file_s = 'C:/Users/ndbke/Dropbox/_NDBK/Research/WHOI/WOA_data/woa18_A5B7_s00_04.nc'

# open NetCDF files for temperature and salinity WOA data
nc_t = xr.open_dataset(file_t, decode_times=False)
nc_s = xr.open_dataset(file_s, decode_times=False)

# get the specific t and s variables out
data_t = nc_t["t_an"]
data_s = nc_s["s_an"]

# isolate the particular parts of data we want
# latitude > 60 deg, the single time value, surface
data_t = data_t.isel(time=0)
data_t = data_t.where(data_t.lat > 60.0)
tem_surface = data_t.isel(depth=0)
print(tem_surface)

data_s = data_s.isel(time=0)
data_s = data_s.where(data_s.lat > 60.0)
sal_surface = data_s.isel(depth=0)
print(sal_surface)

#%%#######################
### plot T, S datasets ###
##########################

plt.figure()
tem_surface.plot(robust=True)  # robust=True removes outliers
plt.ylim(60,90)

plt.figure()
sal_surface.plot(robust=True)
plt.ylim(60,90)
