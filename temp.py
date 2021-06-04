import xarray as xr
import matplotlib.pyplot as plt

file_t = 'C:/Users/ndbke/Dropbox/_NDBK/Research/WHOI/WOA_data/woa18_A5B7_t00_04.nc'
file_s = 'C:/Users/ndbke/Dropbox/_NDBK/Research/WHOI/WOA_data/woa18_A5B7_s00_04.nc'

# open NetCDF files for temperature and salinity WOA data
nc_t = xr.open_dataset(file_t, decode_times=False)
nc_s = xr.open_dataset(file_s, decode_times=False)

# get the specific t and s variables out
data_t = nc_t["t_an"]
data_s = nc_s["s_an"]

# isolate the particular parts of data we want
# latitude > 60 deg, pick the single time value
tem = data_t.isel(time=0)
tem = tem.where(tem.lat > 60.0)
sal = data_s.isel(time=0)
sal = sal.where(sal.lat > 60.0)

# plot both datasets (w/o outliers!)
plt.figure()
tem_surface = tem.isel(depth=0)
tem_surface.plot(robust=True)
plt.ylim(60,90)

plt.figure()
sal_surface = sal.isel(depth=0)
sal_surface.plot(robust=True)
plt.ylim(60,90)



print(tem.coords["lon"])
