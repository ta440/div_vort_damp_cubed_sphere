'''
A script to investigate a grid cell on the sphere,
how the noncoordinate basis vectors are formed,
the computation of the internal angles, 
and the calculation of the area via the spherical excess formula.

Work with the equiangular mapping here.

'''

import numpy as np
from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from functions import *

###########################

# Pick a resolution:
Cn = 192

# Pick indices of the cell
x_index = 6
y_index = 6

# Construct the grid:
R = 6371.220 # Earth's radius in km
a = R/np.sqrt(3)

omega_ref = np.pi/4.
omegas = np.linspace(-omega_ref, omega_ref, Cn + 1)

x = a*np.tan(omegas)
y = a*np.tan(omegas)

xd, yd = np.meshgrid(x,y)
ad = np.ones_like(xd)*a
r = np.sqrt(a**2 + xd**2 + yd**2)

# Perform an analysis on the first tile:
X1, Y1, Z1 = gnomonic_proj(r, R, ad, xd, yd)

print(np.shape(X1))

###############
# Select cell of interest
p1 = np.array([X1[x_index, y_index], Y1[x_index, y_index], Z1[x_index, y_index]])
p2 = np.array([X1[x_index + 1, y_index], Y1[x_index + 1, y_index], Z1[x_index + 1, y_index]])
p3 = np.array([X1[x_index + 1, y_index + 1], Y1[x_index + 1, y_index + 1], Z1[x_index + 1, y_index + 1]])
p4 = np.array([X1[x_index, y_index + 1], Y1[x_index, y_index + 1], Z1[x_index, y_index + 1]])

print(p1, p2, p3, p4)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf1 = ax.scatter(X1[x_index, y_index], Y1[x_index, y_index], Z1[x_index, y_index])
surf2 = ax.scatter(X1[x_index+ 1, y_index], Y1[x_index+ 1, y_index], Z1[x_index+ 1, y_index])
surf3 = ax.scatter(X1[x_index+ 1, y_index+ 1], Y1[x_index+ 1, y_index+ 1], Z1[x_index+ 1, y_index+ 1])
surf4 = ax.scatter(X1[x_index, y_index+ 1], Y1[x_index, y_index+ 1], Z1[x_index, y_index+ 1])
surf5 = ax.plot3D(X1, Y1, Z1)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf1 = ax.scatter(X1[x_index, y_index], Y1[x_index, y_index], Z1[x_index, y_index])
surf2 = ax.scatter(X1[x_index+ 1, y_index], Y1[x_index+ 1, y_index], Z1[x_index+ 1, y_index])
surf3 = ax.scatter(X1[x_index+ 1, y_index+ 1], Y1[x_index+ 1, y_index+ 1], Z1[x_index+ 1, y_index+ 1])
surf4 = ax.scatter(X1[x_index, y_index+ 1], Y1[x_index, y_index+ 1], Z1[x_index, y_index+ 1])

###########################
# Look at point 1 and compute basis vectors

e12 = np.cross(p1, p2)/np.linalg.norm(np.cross(p1, p2))
#e12 = np.cross(p2, p1)/np.linalg.norm(np.cross(p2, p1))

e14 = np.cross(p1, p4)/np.linalg.norm(np.cross(p1, p4))
#e14 = np.cross(p4, p1)/np.linalg.norm(np.cross(p4, p1))


# What is the angle between these?
alpha_412 = np.arccos(np.dot(e12, e14))
print(alpha_412*180/np.pi)

# Convert to degrees also
alpha_123 = alpha_ijk(p1, p2, p3)
alpha_234 = alpha_ijk(p2, p3, p4)
alpha_341 = alpha_ijk(p3, p4, p1)
alpha_412 = alpha_ijk(p4, p1, p2)

tot_angle = alpha_123 + alpha_234 + alpha_341 + alpha_412
cell_area = (R**2)*(tot_angle-2*np.pi)

print(f'Angle 123 in deg is {alpha_123*180/np.pi}')
print(f'Angle 234 in deg is {alpha_234*180/np.pi}')
print(f'Angle 341 in deg is {alpha_341*180/np.pi}')
print(f'Angle 412 in deg is {alpha_412*180/np.pi}')

print(f'Total internal angle is {tot_angle*180/np.pi}')

print(f'Cell area is {cell_area} km^2')



fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.scatter(X1[x_index, y_index], Y1[x_index, y_index], Z1[x_index, y_index], c='k')
ax.scatter(X1[x_index+ 1, y_index], Y1[x_index+ 1, y_index], Z1[x_index+ 1, y_index], c='b')
ax.scatter(X1[x_index+ 1, y_index+ 1], Y1[x_index+ 1, y_index+ 1], Z1[x_index+ 1, y_index+ 1], c='k')
ax.scatter(X1[x_index, y_index+ 1], Y1[x_index, y_index+ 1], Z1[x_index, y_index+ 1], c='r')
ax.quiver(X1[x_index, y_index], Y1[x_index, y_index], Z1[x_index, y_index], e12[0], e12[1], e12[2], length=50, normalize=True, color='b')
ax.quiver(X1[x_index, y_index], Y1[x_index, y_index], Z1[x_index, y_index], e14[0], e14[1], e14[2], length=50, normalize=True, color='r')
ax.quiver(X1[x_index, y_index], Y1[x_index, y_index], Z1[x_index, y_index], p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2], length=50, normalize=True, color='b')


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf5 = ax.plot3D(X1, Y1, Z1)
ax.quiver(X1[x_index, y_index], Y1[x_index, y_index], Z1[x_index, y_index], e12[0], e12[1], e12[2], length=2000, normalize=True, color='r')
ax.quiver(X1[x_index, y_index], Y1[x_index, y_index], Z1[x_index, y_index], e14[0], e14[1], e14[2], length=2000, normalize=True, color='r')

plt.show()