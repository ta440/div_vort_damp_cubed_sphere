'''
Compare damping 
using the oscillation free coefficient 
for each mapping.

Plot amplification factors in the 
k_dx vs x space, so that we examine the damping 
of different harmonics across the panel.

We choose the panel slice so that we cover the 
largest and smallest cells. 
Equiangular: x = 0
Equidistant/equi-edge: x = y

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

# Choose the grid resolution:
C_N = 96

# Choose the order of damping:
q = 4

# Diffusion coefficients for each mapping
# Use the opscillation-free here:

if C_N == 96:
    C_distant = 0.144
    C_angular = 0.117
    C_edge = 0.144

    C_distant_full = 0.109
    C_angular_full = 0.117
    C_edge_full = 0.109

elif C_N == 192:
    C_distant = 0.144
    C_angular = 0.117
    C_edge = 0.144
else:
    # Add your own values in for a different resolution
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
dx_vals_distant, dy_vals_distant, mean_sina_distant, mean_cosa_distant, chi_distant, areas_distant = grid_properties(C_N, R, X1_distant, Y1_distant, Z1_distant)
dx_vals_angular, dy_vals_angular, mean_sina_angular, mean_cosa_angular, chi_angular, areas_angular = grid_properties(C_N, R, X1_angular, Y1_angular, Z1_angular)
dx_vals_edge, dy_vals_edge, mean_sina_edge, mean_cosa_edge, chi_edge, areas_edge = grid_properties(C_N, R, X1_edge, Y1_edge, Z1_edge)

#########################################
# Construct panel index space
x_vals = np.arange(C_N)
x_2d, y_2d = np.meshgrid(x_vals, x_vals)

cent_ind = int(np.floor(C_N/2) - 1)

# Slices of x for the different grids:
inds_diag = np.where(x_2d == y_2d)
inds_centre = np.where(x_2d == np.floor(C_N/2) - 1)


# Construct the normalised wavenumbers:
k_dx = np.linspace(0, np.pi, C_N)

K_DX, X = np.meshgrid(k_dx, x_vals)

# Amplification factor, either from the pseudo-Laplacian
# or the full curvilinear Laplacian
def gamma_along_x(q, C_visc, k_dx, sin_a, cos_a, area, min_area, chi, laplacian_option):
    if laplacian_option == 'pseudo':
        return 1 - ((4*C_visc*min_area*sin_a/area) * (chi*np.sin(k_dx/2)**2 + (1/chi)*np.sin(k_dx/2)**2) )**q
    elif laplacian_option == 'full':
        return 1 - ((4*C_visc*min_area/(area*sin_a)) * (chi*np.sin(k_dx/2)**2 -2*cos_a*np.sin(k_dx/2)*np.sin(k_dx/2)*\
                                                    np.cos(k_dx/2)*np.cos(k_dx/2)+ (1/chi)*np.sin(k_dx/2)**2) )**q


# Compute dampings for the pseudo-Laplacian
distant_damp = np.zeros((C_N, C_N))
angular_damp = np.zeros((C_N, C_N))
edge_damp = np.zeros((C_N, C_N))

# Compute dampings for the full operator
distant_damp_full = np.zeros((C_N, C_N))
angular_damp_full = np.zeros((C_N, C_N))
edge_damp_full = np.zeros((C_N, C_N))

for i in np.arange(C_N):
    distant_damp[i, :] = gamma_along_x(q, C_distant, k_dx,  mean_sina_distant[i,i], mean_cosa_distant[i,i], areas_distant[i,i], np.min(areas_distant), chi_distant[i,i], 'pseudo')
    angular_damp[i, :] =  gamma_along_x(q, C_angular, k_dx, mean_sina_angular[cent_ind, i], mean_cosa_angular[i,i], areas_angular[cent_ind, i], np.min(areas_angular), chi_angular[cent_ind, i], 'pseudo')
    edge_damp[i, :] =  gamma_along_x(q, C_edge, k_dx, mean_sina_edge[i,i], mean_cosa_edge[i,i], areas_edge[i, i], np.min(areas_edge), chi_edge[i,i], 'pseudo')

    distant_damp_full[i, :] = gamma_along_x(q, C_distant_full, k_dx,  mean_sina_distant[i,i], mean_cosa_distant[i,i], areas_distant[i,i], np.min(areas_distant), chi_distant[i,i], 'full')
    angular_damp_full[i, :] =  gamma_along_x(q, C_angular_full, k_dx, mean_sina_angular[cent_ind, i], mean_cosa_angular[i,i],areas_angular[cent_ind, i], np.min(areas_angular), chi_angular[cent_ind, i], 'full')
    edge_damp_full[i, :] =  gamma_along_x(q, C_edge_full, k_dx, mean_sina_edge[i,i], mean_cosa_edge[i,i], areas_edge[i,i], np.min(areas_edge), chi_edge[i,i], 'full')

########################################

cmap_surface = 'jet'

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(K_DX, X, distant_damp, cmap=cmap_surface, linewidth=0, vmin=0, vmax=1)
ax.set_zlim(0, 1)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(K_DX, X, angular_damp, cmap=cmap_surface, linewidth=0, vmin=0, vmax=1)
ax.set_zlim(0, 1)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(K_DX, X, edge_damp, cmap=cmap_surface, linewidth=0, vmin=0, vmax=1)
ax.set_zlim(0, 1)

# All in one.
fig, axes = plt.subplots(1,3, figsize=(18,5.5), constrained_layout=True, subplot_kw={"projection": "3d"})
(ax1,ax2,ax3) = axes
surf1 = ax1.plot_surface(K_DX, X, distant_damp, cmap=cmap_surface, linewidth=0, vmin=0, vmax=1)
surf2 = ax2.plot_surface(K_DX, X, angular_damp, cmap=cmap_surface, linewidth=0, vmin=0, vmax=1)
surf3 = ax3.plot_surface(K_DX, X, edge_damp, cmap=cmap_surface, linewidth=0, vmin=0, vmax=1)

ax1.set_title('Equidistant')
ax2.set_title('Equiangular')
ax3.set_title('Equi-edge')
plt.suptitle('The pseudo-Laplacian')

for ax in axes:
    ax.set_zlim(0, 1)
    ax.set_xlabel(r'$k \Delta x$', labelpad=10)
    ax.set_ylabel(r'$x$ index', labelpad=10)
    ax.set_zlabel(r'$\Gamma$')

# All in one.
fig, axes = plt.subplots(1,3, figsize=(18,5.5), constrained_layout=True, subplot_kw={"projection": "3d"})
(ax1,ax2,ax3) = axes
surf1 = ax1.plot_surface(K_DX, X, distant_damp_full, cmap=cmap_surface, linewidth=0, vmin=0, vmax=1)
surf2 = ax2.plot_surface(K_DX, X, angular_damp_full, cmap=cmap_surface, linewidth=0, vmin=0, vmax=1)
surf3 = ax3.plot_surface(K_DX, X, edge_damp_full, cmap=cmap_surface, linewidth=0, vmin=0, vmax=1)
ax1.set_zlim(0, 1)
ax2.set_zlim(0, 1)
ax3.set_zlim(0, 1)
plt.suptitle('The full Laplacian')

ax1.set_title('Equidistant')
ax2.set_title('Equiangular')
ax3.set_title('Equi-edge')

for ax in axes:
    ax.set_zlim(0, 1)
    ax.set_xlabel(r'$k \Delta x$', labelpad=10)
    ax.set_ylabel(r'$x$ index', labelpad=10)
    ax.set_zlabel(r'$\Gamma$')


plt.show()