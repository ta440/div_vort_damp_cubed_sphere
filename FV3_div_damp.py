# Investigating divergence damping for the FV
# dynamical core, which uses a lon lat grid.
# Here, I use the expressions used in CAM 5.0
# and investigate the default viscosity coefficients.


import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
import matplotlib.colors as colors

#####################################

# Define the grid resolution cubed-sphere
# as number of cells per edge
res = 96

R = 6371.220 # Earth's radius in km

# Default parameters in CAM
C2 = 0.02
C2_sponge = 0.15
C4 = 0.15
C6 = 0.15
C8 = 0.15
dt = 1
R = 6371.220 # Earth's radius in km

# Choose cubed-sphere projection type:
# Option 1 is equidistant
# Option 2 is equiangular
# Option 3 is equi-edge
projection_opt = 2

if projection_opt == 1:
    alpha = 1
    chi = 0.993
    a = R/np.sqrt(3)
    sin_a = 0.8779605
elif projection_opt == 2:
    alpha = np.pi/4
    chi = 1.414
    a = R/np.sqrt(3)
    sin_a = 1.0
elif projection_opt == 3:
    alpha = np.pi/4
    chi = 1.009
    a = R*np.sqrt(2/3)
    sin_a = 0.88129576


#########################################
# Compute the local coordinate cubed-sphere grid

alphas = np.linspace(-alpha, alpha, res+1)

if projection_opt == 1:
    x = a*alphas
    y = a*alphas
else:   
    x = a*np.tan(alphas)
    y = a*np.tan(alphas)

xd, yd = np.meshgrid(x,y)

#########################################

# Investigate the amplification factors in the 
# range of k*dx, l*dy in [0,pi]
k_spacing = 100
k_dx = np.linspace(0, np.pi, k_spacing)
l_dy = np.linspace(0, np.pi, k_spacing)

KDX, LDY = np.meshgrid(k_dx, l_dy)

#########################################

# Define functions for the divergence damping stabilities

def divdamp2_stab(k_dx, l_dy, chi, sin_a, alpha, dt, C2):

    return 1 - 4*dt*C2*sin_a*(chi*(np.sin(k_dx/2)**2) + (1/chi)*(np.sin(l_dy/2)**2))

def divdamp_hyper_stab(k_dx, l_dy, chi, sin_a, alpha, dt, C_2q, q):

    return 1 - (4**q)*dt*(C_2q**q)*(sin_a**q)*(chi*(np.sin(k_dx/2)**2) + (1/chi)*(np.sin(l_dy/2)**2))**q

###########################
# Plotting time!

amp_fact_2 = divdamp2_stab(KDX, LDY, chi, sin_a, alpha, dt, C2)
amp_fact_2_sponge = divdamp2_stab(KDX, LDY, chi, sin_a, alpha, dt, C2_sponge)
amp_fact_4 = divdamp_hyper_stab(KDX, LDY, chi, sin_a, alpha, dt, C4, 2)
amp_fact_6 = divdamp_hyper_stab(KDX, LDY, chi, sin_a, alpha, dt, C6, 3)
amp_fact_8 = divdamp_hyper_stab(KDX, LDY, chi, sin_a, alpha, dt, C8, 4)

norm = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

# 2nd order
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(KDX, LDY, amp_fact_2, cmap='coolwarm',
                       norm=norm, linewidth=0)
ax.set_xlabel('$k \u0394 x$')
ax.set_ylabel('$l \u0394 y$')
ax.set_zlabel('$\u0393_2$')
plt.title('2nd order divergence damping')
surf.cmap.set_under('k')

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(KDX, LDY, amp_fact_2_sponge, cmap='coolwarm',
                       norm=norm, linewidth=0)
ax.set_xlabel('$k \u0394 x$')
ax.set_ylabel('$l \u0394 y$')
ax.set_zlabel('$\u0393_2$')
plt.title('2nd order divergence damping in sponge layer')
surf.cmap.set_under('k')

# 4th order
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(KDX, LDY, amp_fact_4, cmap='coolwarm',
                       norm=norm, linewidth=0)
ax.set_xlabel('$k \u0394 x$')
ax.set_ylabel('$l \u0394 y$')
ax.set_zlabel('$\u0393_4$')
plt.title('4th order divergence damping')
surf.cmap.set_under('k')

# 6th order
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(KDX, LDY, amp_fact_6, cmap='coolwarm',
                       norm=norm, linewidth=0)

ax.set_xlabel('$k \u0394 x$')
ax.set_ylabel('$l \u0394 y$')
ax.set_zlabel('$\u0393_6$')
plt.title('6th order divergence damping')
surf.cmap.set_under('k')

# 8th order
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(KDX, LDY, amp_fact_8, cmap='coolwarm',
                       norm=norm, linewidth=0)
ax.set_xlabel('$k \u0394 x$')
ax.set_ylabel('$l \u0394 y$')
ax.set_zlabel('$\u0393_8$')
plt.title('8th order divergence damping')
surf.cmap.set_under('k')


###################################
# Make plots of the scale selectivity

ndx = 2*np.pi/k_dx

plt.figure()
plt.plot(k_dx, amp_fact_2.diagonal(), label='2nd order')
plt.plot(k_dx, amp_fact_2_sponge.diagonal(), label='2nd order sponge')
plt.plot(k_dx, amp_fact_4.diagonal(), label='4th order')
plt.plot(k_dx, amp_fact_6.diagonal(), label='6th order')
plt.plot(k_dx, amp_fact_8.diagonal(), label='8th order')
plt.legend()


plt.show()