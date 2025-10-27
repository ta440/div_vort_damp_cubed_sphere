'''
Helpful functions for the cubed-sphere mappings.

'''
import numpy as np

def xyz_to_lon_lat(x, y, z):
    # Convert from Cartesian to lon-lat coordinats
    lamda = np.arctan2(y, x)
    phi = np.arcsin(z/np.sqrt(x**2 + y**2 + z**2))
    
    return lamda, phi

def great_circle_dist(lamda_1, lamda_2, phi_1, phi_2, R):
    # Compute the great circle distance in lon-lat coordinates
    return R*np.arccos(np.cos(phi_1)*np.cos(phi_2)*np.cos(lamda_1 - lamda_2) + np.sin(phi_1)*np.sin(phi_2))

def alpha_ijk(p_i, p_j, p_k):
    # Compute the angle between three points,
    # which are given by position vectors
    # in Cartesian space
    e_ji = np.cross(p_j, p_i)
    e_jk = np.cross(p_j, p_k)
    num = np.dot(e_ji, e_jk)
    denom = np.dot(np.linalg.norm(e_ji), np.linalg.norm(e_jk))

    return np.arccos(num/denom)

def gnomonic_proj(r, R, x_vals, y_vals, z_vals):
    # Project the 2D panel coordinates onto the sphere
    # as 3D Cartesian coordinates.
    # R is the radius, r = sqrt(a**2 + x**2 + y**2)
    X = (R/r)*x_vals
    Y = (R/r)*y_vals
    Z = (R/r)*z_vals
    return X, Y, Z

def grid_properties(C_N, R, X, Y, Z):
    dx_vals = np.zeros((C_N+1,C_N))
    dy_vals = np.zeros((C_N,C_N+1))
    areas = np.zeros((C_N,C_N))
    chi = np.zeros((C_N,C_N))
    mean_sina = np.zeros((C_N,C_N))

    LON, LAT = xyz_to_lon_lat(X, Y, Z)

    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)

    for i in np.arange(C_N + 1):
        for j in np.arange(C_N + 1):
            if j != C_N:
                dx_vals[i,j] = great_circle_dist(LON[i,j], LON[i, j+1], LAT[i, j], LAT[i, j+1], R)
            if i != C_N:
                dy_vals[i,j] = great_circle_dist(LON[i,j], LON[i+1, j], LAT[i, j], LAT[i+1, j], R)
            
            #
            if i != C_N and j != C_N:
                # Compute angles in Cartesian coordinates
                point1 = np.array([X[i,j], Y[i,j], Z[i,j]])
                point2 = np.array([X[i,j+1], Y[i,j+1], Z[i,j+1]])
                point3 = np.array([X[i+1,j+1], Y[i+1,j+1], Z[i+1,j+1]])
                point4 = np.array([X[i+1,j], Y[i+1,j], Z[i+1,j]])

                alpha_123 = alpha_ijk(point1, point2, point3)
                alpha_234 = alpha_ijk(point2, point3, point4)
                alpha_341 = alpha_ijk(point3, point4, point1)
                alpha_412 = alpha_ijk(point4, point1, point2)

                mean_sina[i,j] = (np.sin(alpha_123)+np.sin(alpha_234)+np.sin(alpha_341)+np.sin(alpha_412))/4
            
                areas[i,j] = (R**2)*(alpha_123+alpha_234+alpha_341+alpha_412 - 2*np.pi)

    # Compute chi using average dx and dy vals on either side of cell
    for i in np.arange(C_N):
        for j in np.arange(C_N):
            dx_ave = 0.5*(dx_vals[i,j] + dx_vals[i+1,j])
            dy_ave = 0.5*(dy_vals[i,j] + dy_vals[i,j+1])
            chi[i,j] = dy_ave/dx_ave

    return dx_vals, dy_vals, mean_sina, chi, areas