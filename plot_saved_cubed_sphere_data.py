'''
Plots for the paper, using saved data 
from the python script:
'cubed_sphere_projection_all_C_D.py'


'''

'''
A script to read in saved data about the different cubed-sphere
grids and to make plots of these.
'''

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
import matplotlib.colors as colors
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from os.path import abspath, dirname

##########################################
# User-defined parameters:

# Choose grid for storing prognostic at cell centre
grid = 'D'

# Choose the resolution by number of edges
# on each panel of the cubed-sphere
# Typically, set this as 2^n, n integer.
C_N = 96

label_size=14
title_size=14
smaller_size=12

###############################
# For reading in the results:
save_dir = f'{abspath(dirname(__file__))}/saved_arrays'

distant_save_name = f'Equidistant_C{C_N}_{grid}_grid'
angular_save_name = f'Equiangular_C{C_N}_{grid}_grid'
edge_save_name = f'Equi-edge_C{C_N}_{grid}_grid'

###########################
###########################
# Plot of the areas
########
cmap_choice = 'gray_r'

distant_areas = np.load(f'{save_dir}/{distant_save_name}_areas.npy')
angular_areas = np.load(f'{save_dir}/{angular_save_name}_areas.npy')
edge_areas = np.load(f'{save_dir}/{edge_save_name}_areas.npy')

fig, axes = plt.subplots(1,3, figsize=(12,4.5), sharey=True, constrained_layout=True)
(ax1,ax2,ax3) = axes
plot1 = ax1.pcolormesh(distant_areas, cmap = cmap_choice, vmin = np.min(distant_areas), vmax = np.max(distant_areas))
plot2 = ax2.pcolormesh(angular_areas, cmap = cmap_choice,vmin = np.min(angular_areas), vmax = np.max(angular_areas))
plot3 = ax3.pcolormesh(edge_areas, cmap = cmap_choice,vmin = np.min(edge_areas), vmax = np.max(edge_areas))
ax1.set_box_aspect(1)
ax2.set_box_aspect(1)
ax3.set_box_aspect(1)
ax1.set_title(f'Equidistant \n \n Minimum area of {int(np.round(np.min(distant_areas),0))} km$^2$ \n Maximum area of {int(np.round(np.max(distant_areas),0))} km$^2$', size=title_size)
ax2.set_title(f'Equiangular \n \n Minimum area of {int(np.round(np.min(angular_areas),0))} km$^2$ \n Maximum area of {int(np.round(np.max(angular_areas),0))} km$^2$', size=title_size)
ax3.set_title(f'Equi-edge \n \n Minimum area of {int(np.round(np.min(edge_areas),0))} km$^2$ \n Maximum area of {int(np.round(np.max(edge_areas),0))} km$^2$', size=title_size)

cbar1 = plt.colorbar(plot1,ax=ax1,fraction=0.05, pad=0.04)
cbar2 = plt.colorbar(plot2,ax=ax2,fraction=0.05, pad=0.04)
cbar3 = plt.colorbar(plot3,ax=ax3,fraction=0.05, pad=0.04)

#cbar1.set_label('Cell areas (km$^2$)')
#cbar1.set_label('Cell areas (km$^2$)')
cbar3.set_label('Cell areas (km$^2$)', size=smaller_size)

fig.supylabel('y cell index', size=label_size)
fig.supxlabel('x cell index', size=label_size)

###################################

# What about a single colorbar?
cmap_choice = 'jet'

minmin = 1400
maxmax = 3800

conts = np.linspace(minmin,maxmax, 7)
tick_range = np.linspace(minmin,maxmax, 7)
norm = colors.Normalize(vmin=minmin,vmax=maxmax)

fig, axes = plt.subplots(1,3, figsize=(12,5), sharey=True, constrained_layout=True)
(ax1,ax2,ax3) = axes
plot1 = ax1.contourf(distant_areas, cmap = cmap_choice, norm=norm, levels = conts, extend='both')
plot2 = ax2.contourf(angular_areas, cmap = cmap_choice, norm=norm, levels = conts, extend='both')
plot3 = ax3.contourf(edge_areas, cmap = cmap_choice, norm=norm, levels = conts, extend='both')
ax1.set_box_aspect(1)
ax2.set_box_aspect(1)
ax3.set_box_aspect(1)
ax1.set_title(f'Equidistant \n \n Minimum area of {int(np.round(np.min(distant_areas),0))} km$^2$ \n Maximum area of {int(np.round(np.max(distant_areas),0))} km$^2$', size=title_size)
ax2.set_title(f'Equiangular \n \n Minimum area of {int(np.round(np.min(angular_areas),0))} km$^2$ \n Maximum area of {int(np.round(np.max(angular_areas),0))} km$^2$', size=title_size)
ax3.set_title(f'Equi-edge \n \n Minimum area of {int(np.round(np.min(edge_areas),0))} km$^2$ \n Maximum area of {int(np.round(np.max(edge_areas),0))} km$^2$', size=title_size)

cbar = plt.colorbar(plot1, ticks = tick_range, ax=ax3,fraction=0.05, pad=0.04,extend='both')
cbar.set_label('Cell areas (km$^2$)', size=smaller_size)

fig.supylabel('y cell index', size=label_size)
fig.supxlabel('x cell index', size=label_size)

###############################

# What about a single colorbar?
cmap_choice = 'jet'
cmap = plt.colormaps[cmap_choice]
cmap.set_under('black')
cmap.set_over('white')


minmin = 1400
maxmax = 3800

conts = np.linspace(minmin,maxmax, 9)
tick_range = np.linspace(minmin,maxmax, 9)
norm = colors.BoundaryNorm(conts,ncolors=cmap.N)

fig, axes = plt.subplots(1,3, figsize=(12,5), sharey=True, constrained_layout=True)
(ax1,ax2,ax3) = axes
plot1 = ax1.pcolormesh(distant_areas, cmap = cmap, norm=norm)
plot2 = ax2.pcolormesh(angular_areas, cmap = cmap, norm=norm)
plot3 = ax3.pcolormesh(edge_areas, cmap = cmap, norm=norm)
ax1.set_box_aspect(1)
ax2.set_box_aspect(1)
ax3.set_box_aspect(1)
ax1.set_title(f'Equidistant \n \n Minimum area of {int(np.round(np.min(distant_areas),0))} km$^2$ \n Maximum area of {int(np.round(np.max(distant_areas),0))} km$^2$', size=title_size)
ax2.set_title(f'Equiangular \n \n Minimum area of {int(np.round(np.min(angular_areas),0))} km$^2$ \n Maximum area of {int(np.round(np.max(angular_areas),0))} km$^2$', size=title_size)
ax3.set_title(f'Equi-edge \n \n Minimum area of {int(np.round(np.min(edge_areas),0))} km$^2$ \n Maximum area of {int(np.round(np.max(edge_areas),0))} km$^2$', size=title_size)

cbar = plt.colorbar(plot1,ax=ax3,fraction=0.05, pad=0.04,extend='both')
cbar.set_label('Cell areas (km$^2$)', size=smaller_size)

ax1.xaxis.set_major_locator(MultipleLocator(25))
ax2.xaxis.set_major_locator(MultipleLocator(25))
ax3.xaxis.set_major_locator(MultipleLocator(25))

fig.supylabel('y cell index', size=label_size)
fig.supxlabel('x cell index', size=label_size)

###########################
###########################
# Plot of chi
########
cmap_choice = 'seismic'

distant_chi = np.load(f'{save_dir}/{distant_save_name}_chi.npy')
angular_chi = np.load(f'{save_dir}/{angular_save_name}_chi.npy')
edge_chi = np.load(f'{save_dir}/{edge_save_name}_chi.npy')

fig, axes = plt.subplots(1,3, figsize=(12,4.5), sharey=True, constrained_layout=True)
(ax1,ax2,ax3) = axes
plot1 = ax1.pcolormesh(distant_chi, cmap = cmap_choice, norm = colors.TwoSlopeNorm(vmin=np.min(distant_chi), vcenter=1, vmax=np.max(distant_chi)))
plot2 = ax2.pcolormesh(angular_chi, cmap = cmap_choice, norm = colors.TwoSlopeNorm(vmin=np.min(angular_chi), vcenter=1, vmax=np.max(angular_chi)))
plot3 = ax3.pcolormesh(edge_chi, cmap = cmap_choice, norm = colors.TwoSlopeNorm(vmin=np.min(edge_chi), vcenter=1, vmax=np.max(edge_chi)))
ax1.set_box_aspect(1)
ax2.set_box_aspect(1)
ax3.set_box_aspect(1)
ax1.set_title(f'Equidistant \n \n Maximum aspect ratio of {np.round(np.max(distant_chi),3)}', size=title_size)
ax2.set_title(f'Equiangular \n \n Maximum aspect ratio of {np.round(np.max(angular_chi),3)}', size=title_size)
ax3.set_title(f'Equi-edge \n \n Maximum aspect ratio of {np.round(np.max(edge_chi),3)}', size=title_size)

cbar1 = plt.colorbar(plot1,ax=ax1,fraction=0.05, pad=0.04)
cbar2 = plt.colorbar(plot2,ax=ax2,fraction=0.05, pad=0.04)
cbar3 = plt.colorbar(plot3,ax=ax3,fraction=0.05, pad=0.04)

cbar3.set_label('Cell aspect ratio', size=smaller_size)

fig.supylabel('y cell index', size=label_size)
fig.supxlabel('x cell index', size=label_size)

###########################
# What about a single  colorbar?

minmin = 0.70
maxmax = 1.40

conts = np.linspace(minmin,maxmax, 8)
tick_range = np.linspace(minmin,maxmax, 8)
norm = colors.TwoSlopeNorm(vmin=minmin, vcenter=1, vmax=maxmax)

fig, axes = plt.subplots(1,3, figsize=(12,5), sharey=True, constrained_layout=True)
(ax1,ax2,ax3) = axes
plot1 = ax1.pcolormesh(distant_chi, cmap = cmap_choice, norm = norm)
plot2 = ax2.pcolormesh(angular_chi, cmap = cmap_choice, norm = norm)
plot3 = ax3.pcolormesh(edge_chi, cmap = cmap_choice, norm = norm)
ax1.set_box_aspect(1)
ax2.set_box_aspect(1)
ax3.set_box_aspect(1)
ax1.set_title(f'Equidistant \n \n Maximum aspect ratio of {np.round(np.max(distant_chi),3)}', size=title_size)
ax2.set_title(f'Equiangular \n \n Maximum aspect ratio of {np.round(np.max(angular_chi),3)}', size=title_size)
ax3.set_title(f'Equi-edge \n \n Maximum aspect ratio of {np.round(np.max(edge_chi),3)}', size=title_size)

cbar = plt.colorbar(plot1,ticks = tick_range,ax=ax3,fraction=0.05, pad=0.04)
cbar.set_label('Cell aspect ratio', size=smaller_size)

fig.supylabel('y cell index', size=label_size)
fig.supxlabel('x cell index', size=label_size)

###########################
# Plot of the stability function
########
cmap_choice = 'gray_r'

distant_stab = np.load(f'{save_dir}/{distant_save_name}_stab_func.npy')
angular_stab = np.load(f'{save_dir}/{angular_save_name}_stab_func.npy')
edge_stab = np.load(f'{save_dir}/{edge_save_name}_stab_func.npy')

fig, axes = plt.subplots(1,3, figsize=(12,4.5), sharey=True, constrained_layout=True)
(ax1,ax2,ax3) = axes
plot1 = ax1.pcolormesh(distant_stab, cmap = cmap_choice, vmin = np.min(distant_stab), vmax = np.max(distant_stab))
plot2 = ax2.pcolormesh(angular_stab, cmap = cmap_choice,vmin = np.min(angular_stab), vmax = np.max(angular_stab))
plot3 = ax3.pcolormesh(edge_stab, cmap = cmap_choice,vmin = np.min(edge_stab), vmax = np.max(edge_stab))
ax1.set_box_aspect(1)
ax2.set_box_aspect(1)
ax3.set_box_aspect(1)
ax1.set_title(f'Equidistant \n \n Minimum value of \u03a8 is {np.round(np.min(distant_stab),5)}', size=title_size)
ax2.set_title(f'Equiangular \n \n Minimum value of \u03a8 is {np.round(np.min(angular_stab),5)}', size=title_size)
ax3.set_title(f'Equi-edge \n \n Minimum value of \u03a8 is {np.round(np.min(edge_stab),5)}', size=title_size)

cbar1 = plt.colorbar(plot1,ax=ax1,fraction=0.05, pad=0.04)
cbar2 = plt.colorbar(plot2,ax=ax2,fraction=0.05, pad=0.04)
cbar3 = plt.colorbar(plot3,ax=ax3,fraction=0.05, pad=0.04)

cbar3.set_label('Stability function, \u03a8', size=smaller_size)

fig.supylabel('y cell index', size=label_size)
fig.supxlabel('x cell index', size=label_size)

###########################
# Common colorbar
cmap_choice = 'jet'
cmap = plt.colormaps[cmap_choice]
cmap.set_under('white')
#cmap.set_over('black')

minmin = 0.5
maxmax = 1.2

conts = np.linspace(minmin,maxmax, 8)
tick_range = np.linspace(minmin,maxmax, 8)
norm = colors.BoundaryNorm(conts,ncolors=cmap.N)

fig, axes = plt.subplots(1,3, figsize=(12,5), sharey=True, constrained_layout=True)
(ax1,ax2,ax3) = axes
plot1 = ax1.pcolormesh(distant_stab, cmap = cmap, norm=norm)
plot2 = ax2.pcolormesh(angular_stab, cmap = cmap, norm=norm)
plot3 = ax3.pcolormesh(edge_stab, cmap = cmap, norm=norm)
ax1.set_box_aspect(1)
ax2.set_box_aspect(1)
ax3.set_box_aspect(1)
ax1.set_title(f'Equidistant \n \n Minimum value of \u03a8 is {np.round(np.min(distant_stab),3)}', size=title_size)
ax2.set_title(f'Equiangular \n \n Minimum value of \u03a8 is {np.round(np.min(angular_stab),3)}', size=title_size)
ax3.set_title(f'Equi-edge \n \n Minimum value of \u03a8 is {np.round(np.min(edge_stab),3)}', size=title_size)

cbar = plt.colorbar(plot1,ax=ax3,fraction=0.05, pad=0.04, extend='both')
cbar3.set_label('Stability function, \u03a8', size=smaller_size)

fig.supylabel('y cell index', size=label_size)
fig.supxlabel('x cell index', size=label_size)

print('Minimum of each stability function')
print(f'Equidistant: {np.min(distant_stab)}')
print(f'Equiangular: {np.min(angular_stab)}')
print(f'Equi-edge: {np.min(edge_stab)}')

print('Maximum of each stability function')
print(f'Equidistant: {np.round(np.max(distant_stab),5)}')
print(f'Equiangular: {np.round(np.max(angular_stab),5)}')
print(f'Equi-edge: {np.round(np.max(edge_stab),5)}')

plt.show()