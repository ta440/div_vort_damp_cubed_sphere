'''
A plotting script to show the location of blow up
in the Held-Suarez test,
on the equi-edge and equiangular grids.
Here, we plot OMEGA at the lowest
model (pressure) level.
The cubed-sphere panel edges are also plotted
here.
'''
import numpy as np
import scipy
from netCDF4 import Dataset
from matplotlib import pyplot as plt
import argparse
import xarray as xr
import matplotlib.colors as colors
import metpy
import cartopy.crs as ccrs

from tomplot_cubed_sphere import *

# Need to use metpy for fv3!

################################
# User definitions
################################

case1 = 'cam_6_4_050_held_suarez_fv3_C96_divdamp'
case2 = 'cam_6_4_050_held_suarez_fv3_C96_equiangular_divdamp'

nc_file1 = case1 + '.cam.h0i.0001-12-27-01800_sixth_order_blowup_41_steps.regrid.1x1.nc'

nc_file2 = case2 + '.cam.h0i.0001-12-27-01800_fourth_order_blowup_39steps.regrid.1x1.nc'
#nc_file2 = case2 + '.cam.h0i.0001-12-27-01800_sixth_order_blowup_25steps.regrid.1x1.nc'

# Path to where the plots are saved
output_ext = 'grid_comps/'

ext_name = ''#'fourth-order'

# Field to compare
field = 'OMEGA850'

# Time index to compare
t_idx1 = -1
t_idx2 = -1

# Choice of colormap
cmap_choice = 'plasma'

###############################
# Directory stuff
##############################

# Define the base root to the data:
run_base = "/glade/derecho/scratch/timand/"

run_path1 = run_base + case1 + '/run/' + nc_file1
run_path2 = run_base + case2 + '/run/' + nc_file2

output_base = "/glade/u/home/timand/"

output_dir = f'CAM_6_4_050_09012025/plotting/{output_ext}'

output_file = output_base+output_dir

print('set up directory stuff')

##############################
################################
# Extract the data
################################

nc1 = Dataset(run_path1)
nc2 = Dataset(run_path2)

# Dimensions are (time, lev, lat, lon) for 3d vars


time = nc1['time'][:]
lat1 = nc1['lat'][:] 
lon1 = nc1['lon'][:] 
lev1 = nc1['lev'][:]

print(nc1['time'][:])
print(nc2['time'][:])

lat2 = nc2['lat'][:] 
lon2 = nc2['lon'][:] 
lev2 = nc2['lev'][:]

field1 = nc1[field][t_idx1,:]
field2 = nc2[field][t_idx2,:]

LON1, LAT1 = np.meshgrid(lon1, lat1)
LON2, LAT2 = np.meshgrid(lon2, lat2)

###################################


fig, axes = plt.subplots(1,2, figsize = (9,3), sharey=True, constrained_layout=True, subplot_kw={'projection': ccrs.PlateCarree()})
(ax1, ax2) = axes

max_omega = 0.4

conts = np.linspace(-max_omega, max_omega,9)
tick_range = np.linspace(-max_omega, max_omega, 5)

# Add cubed-sphere panel edges:
plot_cubed_sphere_panels(ax1, units='deg', shift_fact=-10, color='white', linewidth=0.5)
plot_cubed_sphere_panels(ax2, units='deg', shift_fact=-10, color='white', linewidth=0.5)

lat_ticks = np.linspace(-90,90,5)
lon_ticks = np.linspace(-180,180,9)

cmap = plt.colormaps[cmap_choice]
cmap.set_under('black')
cmap.set_over('black')

plot1 = ax1.contourf(LON1, LAT1, field1, levels=conts, cmap=cmap, extend='both')
plot2 = ax2.contourf(LON2, LAT2, field2, levels=conts, cmap=cmap, extend='both')

#plot1 = ax1.contourf(LON1, LAT1, field1, cmap=cmap, extend='both')
#plot2 = ax2.contourf(LON2, LAT2, field2, cmap=cmap, extend='both')

ax1.set_aspect('equal')
ax2.set_aspect('equal')

fig.supylabel('Latitude (deg)')
fig.supxlabel('Longitude (deg)')

ax1.set_yticks(lat_ticks)
ax1.set_xticks(lon_ticks)
ax2.set_xticks(lon_ticks)

cb = plt.colorbar(plot2,ax=axes,pad=0.05,shrink=0.75,format='%.1f', ticks=tick_range)
cb.set_label(r"$\omega$ (Pa s$^{-1}$) ")

ax1.set_title(f'Equi-edge')
ax2.set_title(f'Equiangular')

savename = output_file+f'HS1994_compare_grids_blowup_{ext_name}_{field}_with_edges.jpg'

print(f'saving file to {savename}')
plt.savefig(savename, bbox_inches='tight', pad_inches=0.1)
