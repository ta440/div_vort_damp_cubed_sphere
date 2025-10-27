'''
A plotting script to compare divergence damping instabilities
on the equiangular and equi-edge grids, for the JW2006 
baroclinic wave test.
'''

import numpy as np
import scipy
from netCDF4 import Dataset
from matplotlib import pyplot as plt
import argparse
import xarray as xr
import matplotlib.colors as colors
import metpy

# Need to use metpy for fv3!

################################
# User definitions
################################

case1 = 'cam_6_4_050_baro_dry_fv3_C96_L30_equi_edge'
case2 = 'cam_6_4_050_baro_dry_fv3_C96_L30_equi_angular'

nc_file1 = case1 + '.cam.h0i.0001-01-01-00000_6th_div_damp_d4bg_0.185.regrid.1x1.nc'
nc_file2 = case2 + '.cam.h0i.0001-01-01-00000_6th_divdamp_d4bg_0.154_38steps.regrid.1x1.nc'

# Path to where the plots are saved
output_ext = 'grid_comps/'

# Field to compare
field = 'OMEGA850'

# Time index to compare
t_idx1 = 30
t_idx2 = 38

# Choice of colormap
cmap_choice = 'jet'
diff_cmap = 'gray_r'

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

lat2 = nc2['lat'][:] 
lon2 = nc2['lon'][:] 
lev2 = nc2['lev'][:]

field1 = nc1[field][t_idx1,:]
field2 = nc2[field][t_idx2,:]

LON1, LAT1 = np.meshgrid(lon1, lat1)
LON2, LAT2 = np.meshgrid(lon2, lat2)

###################################

fig, axes = plt.subplots(1,2, figsize = (9,3), sharey=True, constrained_layout=True)
(ax1,ax2) = axes

conts = np.linspace(-1.0,1.0,9)
tick_range = np.linspace(-1.0, 1.0, 5)

lon_ticks = np.linspace(0,350,8)

cmap = plt.colormaps[cmap_choice]
cmap.set_under('black')
cmap.set_over('black')

plot1 = ax1.contourf(LON1, LAT1, field1, levels=conts, cmap=cmap, extend='both')
plot2 = ax2.contourf(LON2, LAT2, field2, levels=conts, cmap=cmap, extend='both')

ax1.set_aspect('equal')
ax2.set_aspect('equal')

fig.supylabel('Latitude (deg)')
fig.supxlabel('Longitude (deg)')

ax1.set_xticks(lon_ticks)
ax2.set_xticks(lon_ticks)

cb = plt.colorbar(plot2,ax=axes,pad=0.05,shrink=0.75,format='%.1f', ticks=tick_range)
cb.set_label(r"$\omega$ (Pa s$^{-1}$) ")

ax1.set_title(f'Equi-edge')
ax2.set_title(f'Equiangular')

savename = output_file+f'compare_grids_{field}.jpg'

print(f'saving file to {savename}')
plt.savefig(savename, bbox_inches='tight', pad_inches=0.1)
