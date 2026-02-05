'''
Compare stabiltity of the full and pseudo-Laplacian operators.

Especially, create a plot along a panel edge,
which shows the impact at the panel corners
and middle of panel edges.

'''

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
import matplotlib.colors as colors
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from functions import *

#####################################
# Generate a panel for each mapping
#####################################

# Save the figure?
save_the_figure = True

# Choose the grid resolution:
C_N = 192

# Colour scheme
cmap_choice = 'plasma'

label_size=14
title_size=14
smaller_size=12

##############################
# General grid parameters

global R
R = 6371.220 # Earth's radius in km

a = R/np.sqrt(3)

# Method 1: Equidistant:
alpha_ref_1 = 1
beta1 = 1
a1 = beta1*a

# Method 2: Equi-angular:
alpha_ref_2 = np.pi/4
beta2 = 1
a2 = beta2*a

# Method 3: Equi-edge
alpha_ref_3 = np.arcsin(np.sqrt(1/3))
beta3 = np.sqrt(2)
a3 = beta3*a

# Local mappings for each
alphas_1 = np.linspace(-alpha_ref_1, alpha_ref_1, C_N+1)
alphas_2 = np.linspace(-alpha_ref_2, alpha_ref_2, C_N+1)
alphas_3 = np.linspace(-alpha_ref_3, alpha_ref_3, C_N+1)

x1 = a1*alphas_1
x2 = a2*np.tan(alphas_2)
x3 = a3*np.tan(alphas_3)

xd1, yd1 = np.meshgrid(x1,x1)
xd2, yd2 = np.meshgrid(x2,x2)
xd3, yd3 = np.meshgrid(x3,x3)

ad = np.ones_like(xd1)*a

r1 = np.sqrt(ad**2 + xd1**2 + yd1**2)
r2 = np.sqrt(ad**2 + xd2**2 + yd2**2)
r3 = np.sqrt(ad**2 + xd3**2 + yd3**2)

#####################################
# Perform the gnomonic projection and obtain 
# Cartesian coordinates:
X1_distant, Y1_distant, Z1_distant = gnomonic_proj(r1, R, ad, xd1, yd1)
X1_angular, Y1_angular, Z1_angular = gnomonic_proj(r1, R, ad, xd2, yd2)
X1_edge, Y1_edge, Z1_edge = gnomonic_proj(r1, R, ad, xd3, yd3)

LON_distant, LAT_distant = xyz_to_lon_lat(X1_distant,Y1_distant,Z1_distant)
LON_angular, LAT_angular = xyz_to_lon_lat(X1_angular,Y1_angular,Z1_angular)
LON_edge, LAT_edge = xyz_to_lon_lat(X1_edge,Y1_edge,Z1_edge)

dx_vals_distant, dy_vals_distant, mean_sina_distant, mean_cosa_distant, chi_distant, areas_distant = grid_properties(C_N, R, X1_distant, Y1_distant, Z1_distant)
dx_vals_angular, dy_vals_angular, mean_sina_angular, mean_cosa_angular, chi_angular, areas_angular = grid_properties(C_N, R, X1_angular, Y1_angular, Z1_angular)
dx_vals_edge, dy_vals_edge, mean_sina_edge, mean_cosa_edge, chi_edge, areas_edge = grid_properties(C_N, R, X1_edge, Y1_edge, Z1_edge)

#######################################
# Functions for the grid stability functions
pseudo_stab_distant = areas_distant/(mean_sina_distant*np.min(areas_distant)*(chi_distant + 1/chi_distant))
pseudo_stab_angular = areas_angular/(mean_sina_angular*np.min(areas_angular)*(chi_angular + 1/chi_angular))
pseudo_stab_edge = areas_edge/(mean_sina_edge*np.min(areas_edge)*(chi_edge + 1/chi_edge))

full_stab_distant = mean_sina_distant*areas_distant/(np.min(areas_distant)*(chi_distant + 1/chi_distant))
full_stab_angular = mean_sina_angular*areas_angular/(np.min(areas_angular)*(chi_angular + 1/chi_angular))
full_stab_edge = mean_sina_edge*areas_edge/(np.min(areas_edge)*(chi_edge + 1/chi_edge))

stab_diff_distant = pseudo_stab_distant-full_stab_distant
stab_diff_angular = pseudo_stab_angular-full_stab_angular
stab_diff_edge = pseudo_stab_edge-full_stab_edge

# Confirmed, all changes increase decrease the grid stability
# function, so there are greater stability restrictions
# with the full Laplacian compared to the pseudo-Laplacian
print(np.min(stab_diff_distant))
print(np.min(stab_diff_angular))
print(np.min(stab_diff_edge))

print(np.max(stab_diff_distant))
print(np.max(stab_diff_angular))
print(np.max(stab_diff_edge))


########################################
# Pseudo-Laplacian grid stability function
fig, axes = plt.subplots(1,3, figsize=(12,4.5), sharey=True, constrained_layout=True)
(ax1,ax2,ax3) = axes

cmap_choice = 'viridis'
cmap = plt.colormaps[cmap_choice]

minmin = 0.4
maxmax = 1.2

conts = np.linspace(minmin,maxmax, 9)
norm = colors.BoundaryNorm(conts,ncolors=cmap.N)

plot1 = ax1.pcolormesh(pseudo_stab_distant, cmap = cmap, norm=norm)
plot2 = ax2.pcolormesh(pseudo_stab_angular, cmap = cmap, norm=norm)
plot3 = ax3.pcolormesh(pseudo_stab_edge, cmap = cmap, norm=norm)
ax1.set_box_aspect(1)
ax2.set_box_aspect(1)
ax3.set_box_aspect(1)
ax1.set_title(f'Equidistant\n \n Minimum value of \u03a8 is {np.round(np.min(pseudo_stab_distant),3)}', size=title_size)
ax2.set_title(f'Equiangular\n \n Minimum value of \u03a8 is {np.round(np.min(pseudo_stab_angular),3)}', size=title_size)
ax3.set_title(f'Equi-edge\n \n Minimum value of \u03a8 is {np.round(np.min(pseudo_stab_edge),3)}', size=title_size)

cbar1 = plt.colorbar(plot1,ax=ax1,fraction=0.05, pad=0.04)
cbar2 = plt.colorbar(plot2,ax=ax2,fraction=0.05, pad=0.04)
cbar3 = plt.colorbar(plot3,ax=ax3,fraction=0.05, pad=0.04)

cbar3.set_label('Grid stability function, \u03a8', size=smaller_size)

fig.supylabel('y cell index', size=label_size)
fig.supxlabel('x cell index', size=label_size)

if C_N == 192:
    ax1.yaxis.set_ticks([0,48,96,144,192])
    ax1.xaxis.set_ticks([0,48,96,144,192])
    ax2.xaxis.set_ticks([0,48,96,144,192])
    ax3.xaxis.set_ticks([0,48,96,144,192])

##############################################

# Full-Laplacian grid stability function
fig, axes = plt.subplots(1,3, figsize=(12,4.5), sharey=True, constrained_layout=True)
(ax1,ax2,ax3) = axes

cmap_choice = 'viridis'
cmap = plt.colormaps[cmap_choice]

minmin = 0.4
maxmax = 1.2

conts = np.linspace(minmin,maxmax, 9)
norm = colors.BoundaryNorm(conts,ncolors=cmap.N)

plot1 = ax1.pcolormesh(full_stab_distant, cmap = cmap, norm=norm)
plot2 = ax2.pcolormesh(full_stab_angular, cmap = cmap, norm=norm)
plot3 = ax3.pcolormesh(full_stab_edge, cmap = cmap, norm=norm)
ax1.set_box_aspect(1)
ax2.set_box_aspect(1)
ax3.set_box_aspect(1)
ax1.set_title(f'Equidistant\n \n Minimum value of \u03a8 is {np.round(np.min(full_stab_distant),3)}', size=title_size)
ax2.set_title(f'Equiangular\n \n Minimum value of \u03a8 is {np.round(np.min(full_stab_angular),3)}', size=title_size)
ax3.set_title(f'Equi-edge\n \n Minimum value of \u03a8 is {np.round(np.min(full_stab_edge),3)}', size=title_size)

cbar1 = plt.colorbar(plot1,ax=ax1,fraction=0.05, pad=0.04)
cbar2 = plt.colorbar(plot2,ax=ax2,fraction=0.05, pad=0.04)
cbar3 = plt.colorbar(plot3,ax=ax3,fraction=0.05, pad=0.04)

cbar3.set_label('Grid stability function, \u03a8', size=smaller_size)

fig.supylabel('y cell index', size=label_size)
fig.supxlabel('x cell index', size=label_size)

if C_N == 192:
    ax1.yaxis.set_ticks([0,48,96,144,192])
    ax1.xaxis.set_ticks([0,48,96,144,192])
    ax2.xaxis.set_ticks([0,48,96,144,192])
    ax3.xaxis.set_ticks([0,48,96,144,192])


##################################################

# Plot the difference
'''
# Compare grid stability functions across a panel
fig, axes = plt.subplots(1,3, figsize=(12,4.5), sharey=True, constrained_layout=True)
(ax1,ax2,ax3) = axes
plot1 = ax1.pcolormesh(stab_diff_distant, cmap = 'gray_r', vmin = np.min(stab_diff_distant), vmax = np.max(stab_diff_distant))
plot2 = ax2.pcolormesh(stab_diff_angular, cmap = 'gray_r',vmin = np.min(stab_diff_angular), vmax = np.max(stab_diff_angular))
plot3 = ax3.pcolormesh(stab_diff_edge, cmap = 'gray_r',vmin = np.min(stab_diff_edge), vmax = np.max(stab_diff_edge))
ax1.set_box_aspect(1)
ax2.set_box_aspect(1)
ax3.set_box_aspect(1)
ax1.set_title(f'Equidistant, max diff is {np.round(np.max(stab_diff_distant),3)}', size=title_size)
ax2.set_title(f'Equiangular, max diff is {np.round(np.max(stab_diff_angular),3)}', size=title_size)
ax3.set_title(f'Equi-edge, max diff is {np.round(np.max(stab_diff_edge),3)}', size=title_size)

cbar1 = plt.colorbar(plot1,ax=ax1,fraction=0.05, pad=0.04)
cbar2 = plt.colorbar(plot2,ax=ax2,fraction=0.05, pad=0.04)
cbar3 = plt.colorbar(plot3,ax=ax3,fraction=0.05, pad=0.04)

cbar3.set_label('Difference in grid stability function, \u03a8', size=smaller_size)

fig.supylabel('y cell index', size=label_size)
fig.supxlabel('x cell index', size=label_size)

'''

################################################
# What about a single comparison ?

fig, axes = plt.subplots(2,3, figsize=(12,9), sharey=True, sharex=True, constrained_layout=True)
(ax1,ax2,ax3), (ax4,ax5,ax6) = axes

cmap_choice = 'viridis'
cmap = plt.colormaps[cmap_choice]

minmin = 0.4
maxmax = 1.2

conts = np.linspace(minmin,maxmax, 9)
norm = colors.BoundaryNorm(conts,ncolors=cmap.N)

plot1 = ax1.pcolormesh(pseudo_stab_distant, cmap = cmap, norm=norm)
plot2 = ax2.pcolormesh(pseudo_stab_angular, cmap = cmap, norm=norm)
plot3 = ax3.pcolormesh(pseudo_stab_edge, cmap = cmap, norm=norm)
plot4 = ax4.pcolormesh(full_stab_distant, cmap = cmap, norm=norm)
plot5 = ax5.pcolormesh(full_stab_angular, cmap = cmap, norm=norm)
plot6 = ax6.pcolormesh(full_stab_edge, cmap = cmap, norm=norm)
ax1.set_box_aspect(1)
ax2.set_box_aspect(1)
ax3.set_box_aspect(1)
ax4.set_box_aspect(1)
ax5.set_box_aspect(1)
ax6.set_box_aspect(1)
ax1.set_title(f'Equidistant (Pseudo) \n \n Minimum value of \u03a8 is {np.round(np.min(pseudo_stab_distant),3)}', size=title_size)
ax2.set_title(f'Equiangular (Pseudo)\n \n Minimum value of \u03a8 is {np.round(np.min(pseudo_stab_angular),3)}', size=title_size)
ax3.set_title(f'Equi-edge (Pseudo) \n \n Minimum value of \u03a8 is {np.round(np.min(pseudo_stab_edge),3)}', size=title_size)
ax4.set_title(f'Equidistant (Full) \n \n Minimum value of \u03a8 is {np.round(np.min(full_stab_distant),3)}', size=title_size)
ax5.set_title(f'Equiangular (Full) \n \n Minimum value of \u03a8 is {np.round(np.min(full_stab_angular),3)}', size=title_size)
ax6.set_title(f'Equi-edge (Full) \n \n Minimum value of \u03a8 is {np.round(np.min(full_stab_edge),3)}', size=title_size)


cbar6 = plt.colorbar(plot6,ax=ax6,fraction=0.05, pad=0.04, extend='max')
cbar6.set_label('Grid stability function, \u03a8', size=smaller_size)

ax1.set_ylabel(r"$\xi$ cell index", size=label_size)
ax4.set_ylabel(r"$\xi$ cell index", size=label_size)
fig.supxlabel(r"$\eta$ cell index", size=label_size)

if C_N == 192:
    ax1.yaxis.set_ticks([0,48,96,144,192])
    ax1.xaxis.set_ticks([0,48,96,144,192])
    ax2.xaxis.set_ticks([0,48,96,144,192])
    ax3.xaxis.set_ticks([0,48,96,144,192])
    ax4.yaxis.set_ticks([0,48,96,144,192])
    ax4.xaxis.set_ticks([0,48,96,144,192])
    ax5.xaxis.set_ticks([0,48,96,144,192])
    ax6.xaxis.set_ticks([0,48,96,144,192])

print(save_the_figure)

if save_the_figure:
    plt.savefig(f'figures/compare_grid_stab_diff_laplacians.png',  bbox_inches='tight')


###################################################


# Compare differences along an edge:
fig, axes = plt.subplots(1,3, figsize=(12,4.5), sharey=True, constrained_layout=True)
(ax1,ax2,ax3) = axes
ax1.plot(pseudo_stab_distant[0,:], c='b', label=r"Pseudo: $\widetilde\Psi_{\text{min}}$" f'= {np.round(np.min(pseudo_stab_distant),3)}')
ax1.plot(full_stab_distant[0,:], c='r', label=r"Full: $\Psi_{\text{min}}$" f'= {np.round(np.min(full_stab_distant),3)}')
ax2.plot(pseudo_stab_angular[0,:], c='b', label=r"Pseudo: $\widetilde\Psi_{\text{min}}$" f'= {np.round(np.min(pseudo_stab_angular),3)}')
ax2.plot(full_stab_angular[0,:], c='r', label=r"Full: $\Psi_{\text{min}}$" f'= {np.round(np.min(full_stab_angular),3)}')
ax3.plot(pseudo_stab_edge[0,:], c='b', label=r"Pseudo: $\widetilde\Psi_{\text{min}}$" f'= {np.round(np.min(pseudo_stab_edge),3)}')
ax3.plot(full_stab_edge[0,:], c='r', label=r"Full: $\Psi_{\text{min}}$" f'= {np.round(np.min(full_stab_edge),3)}')    

if C_N == 192:
    ax1.xaxis.set_ticks([0,48,96,144,192])
    ax2.xaxis.set_ticks([0,48,96,144,192])
    ax3.xaxis.set_ticks([0,48,96,144,192])
    ax1.set_xlim([0,192])
    ax2.set_xlim([0,192])
    ax3.set_xlim([0,192])

ax1.set_title('Equidistant', size=title_size)
ax2.set_title('Equiangular', size=title_size)   
ax3.set_title('Equi-edge', size=title_size)   

leg1 = ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.45), fontsize=title_size)
leg2 = ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.45), fontsize=title_size)
leg3 = ax3.legend(loc='lower center', bbox_to_anchor=(0.5, -0.45), fontsize=title_size)

ax1.set_ylabel(r"Grid stability function", size=label_size)
#ax2.set_xlabel(r"$\eta$ index", size=label_size)
ax2.set_xlabel(r"$y$ index", size=label_size)

if save_the_figure:
    plt.savefig(f'figures/full_vs_pseudo_stab_edge.png',  bbox_inches='tight')

# Is the equiangular grid stab function smallest at corner or middle of panel edge?
# It's now slightly smaller at the panel corner!
cent_ind = int(np.floor(C_N/2) - 1)
print(cent_ind)
print(full_stab_angular[0,0])
print(full_stab_angular[cent_ind,0])

#corner_rat = 1/(np.sin(np.sqrt(3)/2))**2
corner_rat = 1/(np.sin(0.866))**2
print('Ratio of grid stability functions at the corner is ', corner_rat)

plt.show()


# Compare grid stability functions along an edge