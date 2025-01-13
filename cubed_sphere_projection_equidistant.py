# Map onto the cubed-sphere.


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
    p2 = np.cross(p_k, p_j)
    num = np.dot(p1,p2)
    denom = np.dot(np.linalg.norm(p1),np.linalg.norm(p2))

    return np.acos(num/denom)

def alpha_ijk_2(p_i, p_j, p_k):
    # Compute the angle between three points,
    # which are given by position vectors
    # in Cartesian space
    # Compute the basis vectors:
    e1 = (p_i - p_j)/np.linalg.norm(p_i - p_j)
    e2 = (p_k - p_j)/np.linalg.norm(p_k - p_j)
    return np.acos(np.dot(e1, e2))

def gnomonic_proj(x_vals,y_vals,z_vals):
    X = (R/r)*x_vals
    Y = (R/r)*y_vals
    Z = (R/r)*z_vals
    return X, Y, Z

###########################################

global R
R = 6371.220 # Earth's radius in km
a = R/np.sqrt(3)

# Equidistant:
alpha_ref = 1

# Grid resolution
C_N = 48

alphas = np.linspace(-alpha_ref, alpha_ref, C_N+1)

x = a*alphas
y = a*alphas

print(np.min(x), np.max(x))

xd, yd = np.meshgrid(x,y)
ad = np.ones_like(xd)*a
print(np.min(ad), np.max(ad))

global r
r = np.sqrt(a**2 + xd**2 + yd**2)

# Perform the gnomonic projection and obtain 
# Cartesian coordinates:
X1, Y1, Z1 = gnomonic_proj(ad, xd, yd)
X2, Y2, Z2 = gnomonic_proj(-xd, ad, yd)
X3, Y3, Z3 = gnomonic_proj(-ad, -xd, yd)
X4, Y4, Z4 = gnomonic_proj(xd, -ad, yd)
X5, Y5, Z5 = gnomonic_proj(-yd, xd, ad)
X6, Y6, Z6 = gnomonic_proj(yd, xd, -ad)

X_1d = np.reshape(X1, (np.size(x)**2, ))
Y_1d = np.reshape(Y1, (np.size(x)**2, ))
Z_1d = np.reshape(Z1, (np.size(x)**2, ))

# Plot the six cubed-sphere faces:
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf1 = ax.plot3D(X1,Y1,Z1)
surf2 = ax.plot3D(X2,Y2,Z2)
surf3 = ax.plot3D(X3,Y3,Z3)
surf4 = ax.plot3D(X4,Y4,Z4)
surf5 = ax.plot3D(X5,Y5,Z5)
surf6 = ax.plot3D(X6,Y6,Z6)
plt.title('The cubed-sphere')

#######################################
# Perform an analysis on the first tile:

# Convert the local Cartesian coordinates to lon-lat
LON, LAT = xyz_to_lon_lat(X1,Y1,Z1)

dx_vals = np.zeros((C_N+1,C_N))
dy_vals = np.zeros((C_N,C_N+1))
areas = np.zeros((C_N,C_N))

X1 = np.asarray(X1)
Y1 = np.asarray(Y1)
Z1 = np.asarray(Z1)

alpha_123s = np.zeros((C_N,C_N))
alpha_234s = np.zeros((C_N,C_N))
alpha_341s = np.zeros((C_N,C_N))
alpha_412s = np.zeros((C_N,C_N))

for i in np.arange(C_N + 1):
    for j in np.arange(C_N + 1):
        if j != C_N:
            dx_vals[i,j] = great_circle_dist(LON[i,j], LON[i, j+1], LAT[i, j], LAT[i, j+1])
            #dx_vals[i,j] = np.abs(xd[i,j+1] - xd[i,j])
        if i != C_N:
            dy_vals[i,j] = great_circle_dist(LON[i,j], LON[i+1, j], LAT[i, j], LAT[i+1, j])
            #dy_vals[i,j] = np.abs(yd[i+1,j] - yd[i,j])
        if i != C_N and j != C_N:
            #point1 = [LON[i,j], LAT[i,j], R]
            #point2 = [LON[i+1,j], LAT[i+1,j], R]
            #point3 = [LON[i+1,j+1], LAT[i+1,j+1], R]
            #point4 = [LON[i,j+1], LAT[i,j+1], R]

            # Counting in clockwise direction:
            #point1 = np.array([X1[i,j], Y1[i,j], Z1[i,j]])
            #point2 = np.array([X1[i,j+1], Y1[i,j+1], Z1[i,j+1]])
            #point3 = np.array([X1[i+1,j+1], Y1[i+1,j+1], Z1[i+1,j+1]])
            #point4 = np.array([X1[i+1,j], Y1[i+1,j], Z1[i+1,j]])

            # Counting in anit-clockwise direction:
            point1 = np.array([X1[i,j], Y1[i,j], Z1[i,j]])
            point2 = np.array([X1[i+1,j], Y1[i+1,j], Z1[i+1,j]])
            point3 = np.array([X1[i+1,j+1], Y1[i+1,j+1], Z1[i+1,j+1]])
            point4 = np.array([X1[i,j+1], Y1[i,j+1], Z1[i,j+1]])

            alpha_123 = alpha_ijk(point1, point2, point3)
            alpha_234 = alpha_ijk(point2, point3, point4)
            alpha_341 = alpha_ijk(point3, point4, point1)
            alpha_412 = alpha_ijk(point4, point1, point2)

            alpha_123s[i,j] = alpha_123
            alpha_234s[i,j] = alpha_234
            alpha_341s[i,j] = alpha_341
            alpha_412s[i,j] = alpha_412

            areas[i,j] = (R**2)*(alpha_123+alpha_234+alpha_341+alpha_412 - 2*np.pi)


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
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')

deg2rad = 180/np.pi

plt.figure()
plt.scatter(LON*deg2rad, LAT*deg2rad)
plt.xlabel('Longitude (deg)')
plt.ylabel('Latitude (deg)')
plt.title('Lon-lat')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')


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

# A rought estimation of aspect ratio, for now.
chi = dy_vals[:,:-1]/dx_vals[:-1,:]

plt.figure()
plt.pcolormesh(chi)
plt.title('Local aspect ratio, chi')
plt.xlabel('x index')
plt.ylabel('y index')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.colorbar()

f_chi = chi + 1/chi

plt.figure()
plt.pcolormesh(f_chi)
plt.title('Function of chi')
plt.xlabel('x index')
plt.ylabel('y index')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.colorbar()

plt.figure()
plt.pcolormesh(areas)
plt.title('Cell areas')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.colorbar()

print('Indices of smallest areas:')
print(np.where(areas-np.min(areas) == 0))


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

# Provide some estimates of restriction for divergence damping.
# Alpha is NOT alpha_ref ... 
sin_a = np.sin(alpha_123s)
A_min = np.min(areas)

stab = areas/(sin_a*A_min*f_chi)
rel_stab = stab-np.min(stab)

plt.figure()
plt.pcolormesh(rel_stab,cmap='coolwarm', vmin=-0, vmax=np.max(rel_stab))
plt.title('Stability measure')
plt.colorbar()

print('Indices of most constraining cell for diffusive stability')
stab_inds = np.where(stab-np.min(stab) == 0)
print(stab_inds)
print(stab_inds[0])

print(f'At most constraining cell, chi = {chi[stab_inds]}, f_chi = {f_chi[stab_inds]}, Ac = {areas[stab_inds[0],stab_inds[1]]}')
print(f'sin(alpha) is {sin_a[stab_inds]}, Min stab is {np.min(stab)}, location is {stab_inds}]')
print(f'Von Neumann stability for 2nd order is {2*np.min(stab)/4}')
print(f'Von Neumann stability for 4th order is {(2**(1/2))*np.min(stab)/4}')
print(f'Von Neumann stability for 6th order is {(2**(1/3))*np.min(stab)/4}')
print(f'Von Neumann stability for 8th order is {(2**(1/4))*np.min(stab)/4}')
print(f'Strong stability (not changing sign) for all orders is {np.min(stab)/4}')


plt.show()