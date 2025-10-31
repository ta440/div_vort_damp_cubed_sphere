'''

This script computes and saves important information
relating to a cubed-sphere grid. 
The user parameters are:
a) Cubed-sphere mapping
b) Resolution
c) D- or C-grid (primary or dual)

'''


import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
import matplotlib.colors as colors
from functions import *

##########################################
# User-defined parameters:

# Choose the grid type.
# Following the FV3 notation.
# 0 is for equi-edge
# 1 is for equidistant
# 2 is for equiangular

grid_type = 2

# Choose grid for storing prognostic at cell centre
grid = 'primary'

# Choose the resolution by number of edges
# on each panel of the cubed-sphere
# Typically, set this as 2^n, n integer.
C_N = 384

###########################################
print('\n')
print(f'Investigating the {grid} grid')
print(f'Resolution of C{C_N}')

R = 6371.220 # Earth's radius in km

a = R/np.sqrt(3)

if grid_type == 0:
    omega_ref = np.arcsin(1/np.sqrt(3))
    gamma = np.sqrt(2)
    print('Equi-edge cubed sphere')
    grid_name = 'Equi-edge'
elif grid_type == 1:
    omega_ref = 1.
    gamma = 1.
    print('Equidistant cubed sphere')
    grid_name = 'Equidistant'
elif grid_type == 2:
    omega_ref = np.pi/4.
    gamma = 1.
    print('Equiangular cubed sphere')
    grid_name = 'Equiangular'
print('\n')

###############################

# Define the coordinates for the primary or offset grid.

# There are C_N x C_N values for the primary grid.
# There are (C_N + 1) x (C_N + 1) values on the offset grid

# The offset grid is extended (by half a grid spacing) to compute
# cell areas and alpha values.

delta_omega = 2*omega_ref/C_N
if grid == 'offset':
    panel_vals = C_N + 1
    # Extend the grid by a half index
    omegas = np.linspace(-omega_ref-delta_omega/2., omega_ref+delta_omega/2., panel_vals + 1)
elif grid == 'primary':
    panel_vals = C_N
    omegas = np.linspace(-omega_ref, omega_ref, panel_vals + 1)
else:
    print('Incorrect grid type')


################################

if grid_type == 1:
    x = gamma*a*omegas
    y = gamma*a*omegas
else:   
    x = gamma*a*np.tan(omegas)
    y = gamma*a*np.tan(omegas)

xd, yd = np.meshgrid(x,y)
ad = np.ones_like(xd)*a

r = np.sqrt(a**2 + xd**2 + yd**2)


#######################################
# Perform an analysis on the first tile:
X1, Y1, Z1 = gnomonic_proj(r, R, ad, xd, yd)

# Convert the local Cartesian coordinates to lon-lat
LON, LAT = xyz_to_lon_lat(X1,Y1,Z1)

X1 = np.asarray(X1)
Y1 = np.asarray(Y1)
Z1 = np.asarray(Z1)

dx_vals, dy_vals, mean_sina, chi, areas, alpha_123s, alpha_234s, alpha_341s, alpha_412s = grid_properties(C_N, R, X1, Y1, Z1, return_alphas=True)

print('Minimum length is ', np.min(dx_vals), ' km')
print('Maximum length is ', np.max(dx_vals), ' km')
print('Mean length is ', np.mean(dx_vals), ' km')

print('Minimum area is ', np.min(areas), ' km^2')
print('Maximum area is ', np.max(areas), ' km^2')
print('Mean area is ', np.mean(areas), ' km^2')

plt.figure()
plt.scatter(xd, yd)
plt.xlabel('xd')
plt.ylabel('yd')
plt.title('Local coordinates')

deg2rad = 180/np.pi

plt.figure()
plt.scatter(LON*deg2rad, LAT*deg2rad)
plt.xlabel('Longitude (deg)')
plt.ylabel('Latitude (deg)')
plt.title('Lon-lat')


plt.figure()
plt.pcolormesh(dx_vals)
plt.title('Distance between adjacent x coordinates')
plt.xlabel('x index')
plt.ylabel('y index')
plt.colorbar()

plt.figure()
plt.pcolormesh(dy_vals)
plt.title('Distance between adjacent y coordinates')
plt.xlabel('x index')
plt.ylabel('y index')
plt.colorbar()

plt.figure()
plt.pcolormesh(chi)
plt.title(f'Local aspect ratio, chi \n {grid_name}, {grid}-grid')
plt.xlabel('x index')
plt.ylabel('y index')
plt.colorbar()

f_chi = chi + 1/chi

plt.figure()
plt.pcolormesh(f_chi)
plt.title('Function of chi')
plt.xlabel('x index')
plt.ylabel('y index')
plt.colorbar()

print('Indices of largest f chi')
print(np.where(f_chi-np.max(f_chi) == 0))

plt.figure()
plt.pcolormesh(areas)
plt.title('Cell areas')
plt.colorbar()

print('Indices of smallest areas:', np.where(areas-np.min(areas) == 0))
# Examine cell areas:
# Make this use the same colorbar for all.
minmin = min(np.min(alpha_123s), np.min(alpha_234s), np.min(alpha_341s), np.min(alpha_412s))
maxmax = max(np.max(alpha_123s), np.max(alpha_234s), np.max(alpha_341s), np.max(alpha_412s))

fig, axes = plt.subplots(2,2, constrained_layout=True)
(ax1, ax2), (ax3,ax4) = axes
p1 = ax1.pcolormesh(alpha_123s, vmin=minmin, vmax=maxmax)
ax1.set_title('alpha 123')
p2 = ax2.pcolormesh(alpha_234s, vmin=minmin, vmax=maxmax)
ax2.set_title('alpha 234')
p3 = ax3.pcolormesh(alpha_341s, vmin=minmin, vmax=maxmax)
ax3.set_title('alpha 341')
p4 = ax4.pcolormesh(alpha_412s, vmin=minmin, vmax=maxmax)
ax4.set_title('alpha 412')

cb = plt.colorbar(p4,ax=axes,pad=0.05,shrink=1,format='%.1f')

minmin = min(np.min(np.sin(alpha_123s)), np.min(np.sin(alpha_234s)), np.min(np.sin(alpha_341s)), np.min(np.sin(alpha_412s)))
maxmax = max(np.max(np.sin(alpha_123s)), np.max(np.sin(alpha_234s)), np.max(np.sin(alpha_341s)), np.max(np.sin(alpha_412s)))

print('Minimum sin(alpha) is ', minmin)

fig, axes = plt.subplots(2,2, constrained_layout=True)
(ax1, ax2), (ax3,ax4) = axes
p1 = ax1.pcolormesh(np.sin(alpha_123s), vmin=minmin, vmax=maxmax)
ax1.set_title('sin(alpha 123)')
p2 = ax2.pcolormesh(np.sin(alpha_234s), vmin=minmin, vmax=maxmax)
ax2.set_title('sin(alpha 234)')
p3 = ax3.pcolormesh(np.sin(alpha_341s), vmin=minmin, vmax=maxmax)
ax3.set_title('sin(alpha 341)')
p4 = ax4.pcolormesh(np.sin(alpha_412s), vmin=minmin, vmax=maxmax)
ax4.set_title('sin(alpha 412)')

cb = plt.colorbar(p4,ax=axes,pad=0.05,shrink=1,format='%.2f')


# Check orthogonality at the centre lines of x=y=0
fig, axes = plt.subplots(2,2, constrained_layout=True)
(ax1, ax2), (ax3,ax4) = axes
p1 = ax1.pcolormesh(np.log(1-np.sin(alpha_123s)))
ax1.set_title('log(1-sin(alpha 123))')
p2 = ax2.pcolormesh(np.log(1-np.sin(alpha_234s)))
ax2.set_title('log(1-sin(alpha 234))')
p3 = ax3.pcolormesh(np.log(1-np.sin(alpha_341s)))
ax3.set_title('log(1-sin(alpha 341))')
p4 = ax4.pcolormesh(np.log(1-np.sin(alpha_412s)))
ax4.set_title('log(1-sin(alpha 412))')
cb = plt.colorbar(p4,ax=axes,pad=0.05,shrink=1,format='%.2f')


##############################################################
# Provide some estimates of restriction for divergence damping.
A_min = np.min(areas)

stab = areas/(mean_sina*A_min*f_chi)
rel_stab = stab-np.min(stab)

plt.figure()
plt.pcolormesh(rel_stab,cmap='coolwarm', vmin=-0, vmax=np.max(rel_stab))
plt.title('Stability measure')
plt.colorbar()

plt.figure()
plt.pcolormesh(rel_stab,cmap='gray_r')
plt.title('Stability measure')
plt.colorbar()

plt.figure()
plt.plot(stab[0,:], label='bottom')
plt.plot(stab[-1,:], label='top')
plt.plot(stab[:,0], label='left')
plt.plot(stab[:,-1], label='right')
plt.legend()
plt.title(f'Stability function over the edges \n {grid_name}, {grid}-grid')

tol=1e-10
stab_inds = np.where(np.abs(stab-np.min(stab))<tol)
print('Indices of most constraining cell for diffusive stability:', stab_inds)

print(f'At most constraining cell, chi = {chi[stab_inds]}, f_chi = {f_chi[stab_inds]}, Ac = {areas[stab_inds[0],stab_inds[1]]}')
print(f'sin(alpha) is {mean_sina[stab_inds]}, Min stab is {np.min(stab)}, location is {stab_inds}]')
print(f'Von Neumann stability for 2nd order is {2*np.min(stab)/4}')
print(f'Von Neumann stability for 4th order is {(2**(1/2))*np.min(stab)/4}')
print(f'Von Neumann stability for 6th order is {(2**(1/3))*np.min(stab)/4}')
print(f'Von Neumann stability for 8th order is {(2**(1/4))*np.min(stab)/4}')
print(f'Strong stability (not changing sign) for all orders is {np.min(stab)/4}')


plt.show()