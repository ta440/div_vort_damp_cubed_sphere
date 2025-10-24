'''

Scale selectivity plots for divergence damping with 
a choice of cubed-sphere mapping.


'''


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

# Choose cubed-sphere projection type:
# Option 1 is equidistant
# Option 2 is equiangular
# Option 0 is equi-edge
# This is following the fv3 naming convention

projection_opt = 0

if projection_opt == 1:
    chi = 1.0
    sin_a = 0.86992878
    grid_name = 'Equidistant'
elif projection_opt == 2:
    chi = 1.40276798
    sin_a = 0.99996708
    grid_name = 'Equiangular'
elif projection_opt == 0:
    chi = 1.0
    sin_a = 0.86992878
    grid_name = 'Equi-edge'

C_val = 'default'

# Divergence damping coefficients:
if C_val == 'default':
    # Default parameters in CAM
    C2 = 0.02
    C2_sponge = 0.15
    C4 = 0.15
    C6 = 0.15
    C8 = 0.15
elif C_val == 'strong':
    if projection_opt == 2:
        C2 = 0.1181712048873376
        C2_sponge = 0.1181712048873376
        C4 = 0.1181712048873376
        C6 = 0.1181712048873376
        C8 = 0.1181712048873376
    else:
        C2 = 0.14368992448310153
        C2_sponge = 0.14368992448310153
        C4 = 0.14368992448310153
        C6 = 0.14368992448310153
        C8 = 0.14368992448310153

#########################################

# Investigate the amplification factors in the 
# range of k*dx, l*dy in [0,pi]
k_spacing = 100
k_dx = np.linspace(0, np.pi, k_spacing)
l_dy = np.linspace(0, np.pi, k_spacing)

KDX, LDY = np.meshgrid(k_dx, l_dy)

#########################################

# Define functions for the divergence damping stabilities

def divdamp2_stab(k_dx, l_dy, chi, sin_a, C2):

    return 1 - 4*C2*sin_a*(chi*(np.sin(k_dx/2)**2) + (1/chi)*(np.sin(l_dy/2)**2))

def divdamp_hyper_stab(k_dx, l_dy, chi, sin_a, C_2q, q):

    return 1 - (4**q)*(C_2q**q)*(sin_a**q)*(chi*(np.sin(k_dx/2)**2) + (1/chi)*(np.sin(l_dy/2)**2))**q

###########################
# Plotting time!
# Here we plot surfaces of the amplification factor 
# in normalised wavenumber space.
# Unstable levels of damping are indicated with black colouring.

label_size=16
title_size=14
smaller_size=12

amp_fact_2 = divdamp2_stab(KDX, LDY, chi, sin_a, C2)
amp_fact_2_sponge = divdamp2_stab(KDX, LDY, chi, sin_a, C2_sponge)
amp_fact_4 = divdamp_hyper_stab(KDX, LDY, chi, sin_a, C4, 2)
amp_fact_6 = divdamp_hyper_stab(KDX, LDY, chi, sin_a, C6, 3)
amp_fact_8 = divdamp_hyper_stab(KDX, LDY, chi, sin_a, C8, 4)

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


###############################################
# Make plots of the scale selectivity
# We take a slice along the diagonal in wavenumber space
# with k delta x = l delta y.

line_width = 2

ndx = 2*np.pi/k_dx

plt.figure(figsize=(8,6))
#plt.plot(k_dx, amp_fact_2.diagonal(), label='2nd order')
plt.plot(k_dx, amp_fact_2_sponge.diagonal(), label='2nd order sponge', c='b', linewidth=line_width)
plt.plot(k_dx, amp_fact_4.diagonal(), label='4th order', c='g', linestyle='dashed', linewidth=line_width)
plt.plot(k_dx, amp_fact_6.diagonal(), label='6th order', c='r', linestyle='dotted', linewidth=line_width)
plt.plot(k_dx, amp_fact_8.diagonal(), label='8th order', c='orange', linestyle='dashdot', linewidth=line_width)
#plt.title(f'{grid_name}, default CAM divergence damping coefficients')
plt.plot(k_dx, np.zeros_like(k_dx), linestyle='--', c='k')
plt.plot(k_dx, -np.ones_like(k_dx), linestyle='-', c='k')
plt.xlabel('Normalised wavenumber, k\u0394x', size=16)
plt.ylabel('Amplification factor', size=16)
plt.xlim(0,np.pi)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14,loc='lower left',framealpha=1)
ax = plt.gca()
ax.xaxis.set_major_formatter(plt.FuncFormatter(
   lambda val,pos: '{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'
))
ax.xaxis.set_major_locator(plt.MultipleLocator(base=0.1*np.pi))

plt.figure(figsize=(8,6))
plt.plot(k_dx, amp_fact_2.diagonal(), label='2nd order', c='b', linewidth=line_width)
plt.plot(k_dx, amp_fact_4.diagonal(), label='4th order', c='g', linestyle='dashed', linewidth=line_width)
plt.plot(k_dx, amp_fact_6.diagonal(), label='6th order', c='r', linestyle='dotted', linewidth=line_width)
plt.plot(k_dx, amp_fact_8.diagonal(), label='8th order', c='orange', linestyle='dashdot', linewidth=line_width)
#plt.title(f'{grid_name}, divergence damping for stronger stability divergence damping')
plt.plot(k_dx, np.zeros_like(k_dx), linestyle='--', c='k')
plt.xlabel('Normalised wavenumber, k\u0394x', size=title_size)
plt.ylabel('Amplification factor \n', size=title_size)
plt.xlim(0,np.pi)
plt.xticks(fontsize=smaller_size)
plt.yticks(fontsize=smaller_size)
plt.legend(fontsize=smaller_size)
ax = plt.gca()
ax.xaxis.set_major_formatter(plt.FuncFormatter(
   lambda val,pos: '{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'
))
ax.xaxis.set_major_locator(plt.MultipleLocator(base=0.1*np.pi))


plt.show()