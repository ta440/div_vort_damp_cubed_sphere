# Map onto the cubed-sphere.
# Here, we compare grids from three different methods:
# 1. Equidistant
# 2. Equiangular
# 3. Equi-edge


import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
import matplotlib.colors as colors

##########################################

# Helper functions:
def xyz_to_lon_lat(x, y, z):
    lamda = np.atan2(y, x)
    phi = np.asin(z/np.sqrt(x**2 + y**2 + z**2))
    
    return lamda, phi

def great_circle_dist(lamda_1, lamda_2, phi_1, phi_2):
    return R*np.acos(np.cos(phi_1)*np.cos(phi_2)*np.cos(lamda_1 - lamda_2) + np.sin(phi_1)*np.sin(phi_2))

def alpha_ijk(p_i, p_j, p_k):
    # Compute the angle between three points,
    # which are given by position vectors
    # in Cartesian space
    p1 = np.cross(p_i, p_j)
    p2 = np.cross(p_j, p_k)
    num = np.dot(p1,p2)
    denom = np.dot(np.linalg.norm(p1),np.linalg.norm(p2))

    # Sign hack for now ...
    return np.acos(-num/denom)

def alpha_ijk_points(p_i, p_j, p_k):
    # Compute the angle between three points,
    # which are given by position vectors
    # in Cartesian space
    A = p_i - p_j
    B = p_k - p_j
    num = np.dot(A,B)
    denom = np.linalg.norm(A)*np.linalg.norm(B)

    return np.acos(num/denom)

def gnomonic_proj(r, x_vals, y_vals, z_vals):
    X = (R/r)*x_vals
    Y = (R/r)*y_vals
    Z = (R/r)*z_vals
    return X, Y, Z

###########################################

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
alpha_ref_3 = np.asin(np.sqrt(1/3))
beta3 = np.sqrt(2)
a3 = beta3*a

# Grid resolution
C_N = 4

######################
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
X1_1, Y1_1, Z1_1 = gnomonic_proj(r1, ad, xd1, yd1)
X2_1, Y2_1, Z2_1 = gnomonic_proj(r1, -xd1, ad, yd1)
X3_1, Y3_1, Z3_1 = gnomonic_proj(r1, -ad, -xd1, yd1)
X4_1, Y4_1, Z4_1 = gnomonic_proj(r1, xd1, -ad, yd1)
X5_1, Y5_1, Z5_1 = gnomonic_proj(r1, -yd1, xd1, ad)
X6_1, Y6_1, Z6_1 = gnomonic_proj(r1, yd1, xd1, -ad)

X1_2, Y1_2, Z1_2 = gnomonic_proj(r2, ad, xd2, yd2)
X2_2, Y2_2, Z2_2 = gnomonic_proj(r2, -xd2, ad, yd2)
X3_2, Y3_2, Z3_2 = gnomonic_proj(r2, -ad, -xd2, yd2)
X4_2, Y4_2, Z4_2 = gnomonic_proj(r2, xd2, -ad, yd2)
X5_2, Y5_2, Z5_2 = gnomonic_proj(r2, -yd2, xd2, ad)
X6_2, Y6_2, Z6_2 = gnomonic_proj(r2, yd2, xd2, -ad)

X1_3, Y1_3, Z1_3 = gnomonic_proj(r3, ad, xd3, yd3)
X2_3, Y2_3, Z2_3 = gnomonic_proj(r3, -xd3, ad, yd3)
X3_3, Y3_3, Z3_3 = gnomonic_proj(r3, -ad, -xd3, yd3)
X4_3, Y4_3, Z4_3 = gnomonic_proj(r3, xd3, -ad, yd3)
X5_3, Y5_3, Z5_3 = gnomonic_proj(r3, -yd3, xd3, ad)
X6_3, Y6_3, Z6_3 = gnomonic_proj(r3, yd3, xd3, -ad)


# Plot the six cubed-sphere faces:
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf1 = ax.plot3D(X1_1,Y1_1,Z1_1)
surf2 = ax.plot3D(X2_1,Y2_1,Z2_1)
surf3 = ax.plot3D(X3_1,Y3_1,Z3_1)
surf4 = ax.plot3D(X4_1,Y4_1,Z4_1)
surf5 = ax.plot3D(X5_1,Y5_1,Z5_1)
surf6 = ax.plot3D(X6_1,Y6_1,Z6_1)
plt.title('Equidistant')

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf1 = ax.plot3D(X1_2,Y1_2,Z1_2)
surf2 = ax.plot3D(X2_2,Y2_2,Z2_2)
surf3 = ax.plot3D(X3_2,Y3_2,Z3_2)
surf4 = ax.plot3D(X4_2,Y4_2,Z4_2)
surf5 = ax.plot3D(X5_2,Y5_2,Z5_2)
surf6 = ax.plot3D(X6_2,Y6_2,Z6_2)
plt.title('Equiangular')

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf1 = ax.plot3D(X1_3,Y1_3,Z1_3)
surf2 = ax.plot3D(X2_3,Y2_3,Z2_3)
surf3 = ax.plot3D(X3_3,Y3_3,Z3_3)
surf4 = ax.plot3D(X4_3,Y4_3,Z4_3)
surf5 = ax.plot3D(X5_3,Y5_3,Z5_3)
surf6 = ax.plot3D(X6_3,Y6_3,Z6_3)
plt.title('Equi-edge')

#######################################
# Perform an analysis on the first tile
# of each

plt.figure()
plt.scatter(xd1, yd1)
plt.xlabel('xd')
plt.ylabel('yd')
plt.title('Equidistant local coordinates')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')

plt.figure()
plt.scatter(xd2, yd2)
plt.xlabel('xd')
plt.ylabel('yd')
plt.title('Equiangular local coordinates')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')

plt.figure()
plt.scatter(xd3, yd3)
plt.xlabel('xd')
plt.ylabel('yd')
plt.title('Equi-edge local coordinates')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')


deg2rad = 180/np.pi

# What about in lat-lon?
LON_1, LAT_1 = xyz_to_lon_lat(X1_1,Y1_1,Z1_1)
LON_2, LAT_2 = xyz_to_lon_lat(X1_2,Y1_2,Z1_2)
LON_3, LAT_3 = xyz_to_lon_lat(X1_3,Y1_3,Z1_3)

plt.figure()
plt.scatter(LON_1*deg2rad, LAT_1*deg2rad)
plt.xlabel('Longitude (deg)')
plt.ylabel('Latitude (deg)')
plt.title('Equidistant panel 1: Lon-lat')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')

plt.figure()
plt.scatter(LON_2*deg2rad, LAT_2*deg2rad)
plt.xlabel('Longitude (deg)')
plt.ylabel('Latitude (deg)')
plt.title('Equiangular panel 1: Lon-lat')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')

plt.figure()
plt.scatter(LON_3*deg2rad, LAT_3*deg2rad)
plt.xlabel('Longitude (deg)')
plt.ylabel('Latitude (deg)')
plt.title('Equi-edge panel 1: Lon-lat')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')


# What about putting all this information into a combined figure?
label_size=16


fig = plt.figure(constrained_layout=True)
(subfig1, subfig2) = fig.subfigures(2,1)
(ax1,ax2,ax3) = subfig1.subplots(1,3, sharey=True)
(ax4,ax5,ax6) = subfig2.subplots(1,3, sharey=True)
ax1.scatter(xd1, yd1)
ax2.scatter(xd2, yd2)
ax3.scatter(xd3, yd3)
ax4.scatter(LON_1*deg2rad, LAT_1*deg2rad)
ax5.scatter(LON_2*deg2rad, LAT_2*deg2rad)
ax6.scatter(LON_3*deg2rad, LAT_3*deg2rad)

ax1.set_box_aspect(1)
ax2.set_box_aspect(1)
ax3.set_box_aspect(1)
ax4.set_box_aspect(1)
ax5.set_box_aspect(1)
ax6.set_box_aspect(1)

subfig1.suptitle('Local cubed-sphere panel coordinate')
ax1.set_title('Equidistant')
ax2.set_title('Equiangular')
ax3.set_title('Equi-edge')

ax1.set_ylabel('$y_d$')
ax2.set_xlabel('$x_d$')

subfig2.suptitle('Longitude-latitude coordinates')

ax4.set_ylabel('Latitiude (degrees)')
ax5.set_xlabel('Longtiude (degrees)')



fig, axes = plt.subplots(1,3, figsize=(12,5), sharey=True)
(ax1,ax2,ax3) = axes
ax1.scatter(xd1, yd1)
ax2.scatter(xd2, yd2)
ax3.scatter(xd3, yd3)
ax1.set_box_aspect(1)
ax2.set_box_aspect(1)
ax3.set_box_aspect(1)
ax1.set_title('Equidistant')
ax2.set_title('Equiangular')
ax3.set_title('Equi-edge')

fig.supylabel('Local y coordinate', size=label_size)
fig.supxlabel('Local x coordinate', size=label_size)

# What about putting all this information into a combined figure?
fig, axes = plt.subplots(1,3, sharey=True)
(ax1,ax2,ax3) = axes
ax1.scatter(LON_1*deg2rad, LAT_1*deg2rad)
ax2.scatter(LON_2*deg2rad, LAT_2*deg2rad)
ax3.scatter(LON_3*deg2rad, LAT_3*deg2rad)

ax1.set_box_aspect(1)
ax2.set_box_aspect(1)
ax3.set_box_aspect(1)
ax1.set_title('Equidistant')
ax2.set_title('Equiangular')
ax3.set_title('Equi-edge')

fig.supylabel('Latitude (degrees)', size=label_size)
fig.supxlabel('Longitude (degrees)', size=label_size)

plt.show()