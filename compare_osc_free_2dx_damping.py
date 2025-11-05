'''
Compare damping over the cubed-sphere
using the oscillation free coefficient 
for each mapping.
Use the primary grid here.
'''


import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
import matplotlib.colors as colors
from functions import *

#####################################
# Generate a panel for each mapping
#####################################

# Save the figure?
save_the_figure = True

# Choose the grid resolution:
C_N = 192

# Choose the order of damping:
q = 2

# Oscillation-free coefficients
C_distant = 0.144
C_angular = 0.117
C_edge = 0.144

#####################################
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

r1= np.sqrt(ad**2 + xd1**2 + yd1**2)
r2= np.sqrt(ad**2 + xd2**2 + yd2**2)
r3= np.sqrt(ad**2 + xd3**2 + yd3**2)

# Perform the gnomonic projection and obtain 
# Cartesian coordinates:
X1_distant, Y1_distant, Z1_distant = gnomonic_proj(r1, R, ad, xd1, yd1)
X1_angular, Y1_angular, Z1_angular = gnomonic_proj(r1, R, ad, xd2, yd2)
X1_edge, Y1_edge, Z1_edge = gnomonic_proj(r1, R, ad, xd3, yd3)

LON_distant, LAT_distant = xyz_to_lon_lat(X1_distant,Y1_distant,Z1_distant)
LON_angular, LAT_angular = xyz_to_lon_lat(X1_angular,Y1_angular,Z1_angular)
LON_edge, LAT_edge = xyz_to_lon_lat(X1_edge,Y1_edge,Z1_edge)

# Compute cell areas and aspect ratios for the grids:
dx_vals_distant, dy_vals_distant, mean_sina_distant, chi_distant, areas_distant = grid_properties(C_N, R, X1_distant, Y1_distant, Z1_distant)
dx_vals_angular, dy_vals_angular, mean_sina_angular, chi_angular, areas_angular = grid_properties(C_N, R, X1_angular, Y1_angular, Z1_angular)
dx_vals_edge, dy_vals_edge, mean_sina_edge, chi_edge, areas_edge = grid_properties(C_N, R, X1_edge, Y1_edge, Z1_edge)

#########################################
# Construct panel index space
x_vals = np.arange(C_N)
x_2d, y_2d = np.meshgrid(x_vals, x_vals)

def two_dx_damp(q, C_visc, sin_a, areas, chis):
    return 1 - ((4*C_visc*np.min(areas)*sin_a/areas) * (chis + 1/chis) )**q

# Compute the 2dx wave damping for each cell
distant_2dx_damp =  two_dx_damp(q, C_distant, mean_sina_distant, areas_distant, chi_distant)
angular_2dx_damp =  two_dx_damp(q, C_angular, mean_sina_angular, areas_angular, chi_angular)
edge_2dx_damp =  two_dx_damp(q, C_edge, mean_sina_edge, areas_edge, chi_edge)

########################################

cmap_surface = 'jet'
big_size=18
smaller_size=18
tick_size = 14


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(x_2d, y_2d, distant_2dx_damp, cmap=cmap_surface, linewidth=0, vmin=0, vmax=1)
ax.set_zlim(0, 1)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(x_2d, y_2d, angular_2dx_damp, cmap=cmap_surface, linewidth=0, vmin=0, vmax=1)
ax.set_zlim(0, 1)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(x_2d, y_2d, edge_2dx_damp, cmap=cmap_surface, linewidth=0, vmin=0, vmax=1)
ax.set_zlim(0, 1)

# All in one.
fig, axes = plt.subplots(1,3, figsize=(18,6), subplot_kw={"projection": "3d"})
(ax1,ax2,ax3) = axes
surf1 = ax1.plot_surface(x_2d, y_2d, distant_2dx_damp, cmap=cmap_surface, linewidth=0, vmin=0, vmax=1)
surf2 = ax2.plot_surface(x_2d, y_2d, angular_2dx_damp, cmap=cmap_surface, linewidth=0, vmin=0, vmax=1)
surf3 = ax3.plot_surface(x_2d, y_2d, edge_2dx_damp, cmap=cmap_surface, linewidth=0, vmin=0, vmax=1)

for ax in axes:
    ax.set_zlim(0, 1)
    ax.set_xlim(0, C_N)
    ax.set_ylim(0, C_N)

    if C_N == 192:
        ax.set_xticks([0,48,96,144,192])
        ax.set_yticks([0,48,96,144,192])

    ax.tick_params(labelsize=tick_size)

    ax.set_xlabel(f'$x$ index', size=smaller_size, labelpad=10)
    ax.set_ylabel(f'$y$ index', size=smaller_size, labelpad=10)
    ax.set_zlabel('$\Gamma(\pi, \pi)$', size=smaller_size, labelpad=10)

ax1.set_title(f'Equidistant \n max($\Gamma(\pi, \pi)$) = {np.max(distant_2dx_damp):.3f}', size=big_size)
ax2.set_title(f'Equiangular \n max($\Gamma(\pi, \pi)$) = {np.max(angular_2dx_damp):.3f}', size=big_size)
ax3.set_title(f'Equi-edge \n max($\Gamma(\pi, \pi)$) = {np.max(edge_2dx_damp):.3f}', size=big_size)

plt.subplots_adjust(wspace=0.3, hspace=0.2, left=0.05, right=0.95)

# Save this figure:
if save_the_figure:
    savename = f'figures/two_dx_damp_over_panel_q{q}.jpg'

    print(f'saving file to {savename}')
    plt.savefig(savename)

# Alternatively, plot flat:
fig, axes = plt.subplots(1,3, figsize=(12,5), sharey=True, constrained_layout=True)
(ax1,ax2,ax3) = axes
conts = np.linspace(0,1,11)
norm = colors.Normalize(vmin=0,vmax=1)
cmap = 'jet'
plot1 = ax1.contourf(x_2d, y_2d, distant_2dx_damp, cmap=cmap, levels=conts, norm=norm)
plot2 = ax2.contourf(x_2d, y_2d, angular_2dx_damp, cmap=cmap, levels=conts, norm=norm)
plot3 = ax3.contourf(x_2d, y_2d, edge_2dx_damp, cmap=cmap, levels=conts, norm=norm)
ax1.set_aspect(1)
ax2.set_aspect(1)
ax3.set_aspect(1)
cbar = plt.colorbar(plot3, ticks = conts, ax=ax3,fraction=0.05, pad=0.05)

print(np.min(distant_2dx_damp), np.max(distant_2dx_damp))
print(np.min(angular_2dx_damp), np.max(angular_2dx_damp))
print(np.min(edge_2dx_damp), np.max(edge_2dx_damp))

plt.show()