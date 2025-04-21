'''
Helpful functions for the cubed-sphere mappings

'''
import numpy as np

def xyz_to_lon_lat(x, y, z):
    # Convert from Cartesian to lon-lat
    lamda = np.atan2(y, x)
    phi = np.asin(z/np.sqrt(x**2 + y**2 + z**2))
    
    return lamda, phi

def great_circle_dist(lamda_1, lamda_2, phi_1, phi_2, R):
    # Compute the great circle distance in lon-lat coordinates
    return R*np.acos(np.cos(phi_1)*np.cos(phi_2)*np.cos(lamda_1 - lamda_2) + np.sin(phi_1)*np.sin(phi_2))

def alpha_ijk(p_i, p_j, p_k):
    # Compute the angle between three points,
    # which are given by position vectors
    # in Cartesian space
    e_ji = np.cross(p_j, p_i)
    e_jk = np.cross(p_j, p_k)
    num = np.dot(e_ji, e_jk)
    denom = np.dot(np.linalg.norm(e_ji),np.linalg.norm(e_jk))

    return np.acos(num/denom)

def gnomonic_proj(r, x_vals, y_vals, z_vals, R):
    # Project the 2D panel coordinates onto the sphere
    # as 3D Cartesian coordinates
    X = (R/r)*x_vals
    Y = (R/r)*y_vals
    Z = (R/r)*z_vals
    return X, Y, Z