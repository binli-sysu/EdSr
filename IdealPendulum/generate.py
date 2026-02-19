#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# get_ipython().run_line_magic('matplotlib', 'Qt6')
import logging
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from idealPendulum import IdealPendulum

fontsize = font_manager.FontProperties(size = 15)
tick_fontsize = font_manager.FontProperties(size = 15)
title_fontsize = font_manager.FontProperties(size = 15.5)
legned_fontsize = font_manager.FontProperties(size = 5.5)


logger = logging.getLogger('global')
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    fmt = '%(asctime)s - %(filename)s - %(levelname)s: %(message)s'
))

logger.addHandler(handler)


# In[ ]:


fig = plt.figure(figsize=[10,4], dpi=200)   


# In[ ]:


tStart     = 0.
tStep      = 10_000_000
interval   = 1
basis_timestep = 0.01
XmaxIter    = 40
VmaxIter   = 40

label_interval = 5
label_basis_timestep = basis_timestep / label_interval
label_length = tStep * interval
# print(times)

mass     = 1.0
radius   = 1.0
gravity  = 1.0
mgr      = mass * gravity * radius

theta = np.pi / 3
# init_state = np.array([mass, radius * np.sin(theta), 0.], dtype = np.float32)
init_state = np.array([mass, theta, 0.], dtype = np.float32)

def potential_energy(orbit):
    potential = mgr * (1 - np.cos(orbit[:, 1]))
    return potential

def kinetic_energy(orbit):
    kinetic = 0.5 * mass * radius**2 * orbit[:, 2] ** 2
    return kinetic

def hamiltonian(orbit):
    hamiltonian = kinetic_energy(orbit) + potential_energy(orbit)
    return hamiltonian

def lagrangian(orbit):
    lagrangian = kinetic_energy(orbit) - potential_energy(orbit)
    return lagrangian

model = IdealPendulum(mass = mass, radius = radius, gravity = gravity)

logger.info(f"start to generate {tStep:,} data.")
# label_orbit, label_times = model.makelabel(init_state, label_basis_timestep, label_interval, label_length)

# orbit, run_times = model.TrajectoryWithVV(init_state, interval * basis_timestep, tStep)

orbit, run_times = model.generateInParallel(init_state, basis_timestep, interval, length = tStep, XmaxIter = XmaxIter, VmaxIter = VmaxIter)
logger.info(f"{tStep:,} data have been generated.")


# In[ ]:


# x: np.ndarray = orbit[..., 1].copy()
# v: np.ndarray = orbit[..., 2].copy()

# orbit[..., 1] = np.arcsin(angle / radius)
# orbit[..., 2] = v * np.sqrt(1 - (angle / radius)**2)

# theta = np.arcsin(x / radius)
# omega = v * np.sqrt(1 - (x / radius)**2) / radius

# orbit[..., 1] = theta
# orbit[..., 2] = omega

theta = orbit[..., 1]
omega = orbit[..., 2]

theta_min = theta.min()
theta_max = theta.max()
omega_min = omega.min()
omega_max = omega.max()

if tStep > 1e5:
    np.savez(f'trajs_n{tStep}_b{basis_timestep}_intv{interval}_iter{XmaxIter}.npz',
        orbit = orbit, times = run_times
    )

plt.subplot(1, 2, 1)


qscale = 0.5

# coord initialization
plt.xlabel(r'$\mathbf{\theta}$', fontsize = fontsize.get_size()) ; plt.ylabel(r'$\mathbf{\omega}$', fontsize = fontsize.get_size(), rotation = 0)
plt.axis()
plt.xlim(theta_min - qscale * abs(theta_min), theta_max + qscale * abs(theta_max))
plt.ylim(omega_min - qscale * abs(omega_min), omega_max + qscale * abs(omega_max))
plt.title('Trajectories', fontsize = title_fontsize.get_size(), fontweight = 'bold')

plt.plot(theta, omega, markersize = 1, alpha = 0.7)
plt.scatter([orbit[0, 1]], [orbit[0, 2]], linewidths = 1, c = 'black', alpha = 1.0)
plt.arrow(orbit[0, 1], orbit[0, 2], (orbit[1, 1] - orbit[0, 1])* 20, (orbit[1, 2] - orbit[0, 2])*20, color = 'black', linewidth = 2)
# sub1_line1, = plt.plot(x, y, 'o', animated = True, markersize = 3)
logger.info(orbit.shape)

plt.legend(['hamiltonian path', 'initial point'],fontsize = 8)


# In[ ]:


potential = potential_energy(orbit)
kinetic = kinetic_energy(orbit)
hamilton = hamiltonian(orbit)
lag = lagrangian(orbit)

times = run_times
logger.info(f'{potential.shape}, {kinetic.shape}, {hamilton.shape}, {times.shape}')
pe, ke, tote, t = [], [], [], []

plt.subplot(1, 2, 2)
plt.plot(times, -potential, 'o', markersize = 1)
plt.plot(times, kinetic, 'o', markersize = 1)
plt.plot(times, hamilton, 'o', markersize = 1, alpha = 0.7)


# plt.plot(times, lag, 'o', markersize = 2)
tmin, tmax = times.min(), times.max()

e_min = min((-potential).min(), kinetic.min(), hamilton.min(), -1e-1)
e_max = max((-potential).max(), kinetic.max(), hamilton.max())

logger.info(f'min energy: {e_min}\nmax energy: {e_max}')

# frames = np.stack([x, y, potential, kinetic, total, times], axis = 1)
# timesteps = 1
# frames = frames[::timesteps]

escale = 0.1

# energy initialization
plt.axis()
plt.title("Energy", fontsize = title_fontsize.get_size(), fontweight = 'bold')
plt.xlabel('time', fontsize = fontsize.get_size(), fontweight = 'bold')
plt.xscale('symlog')
plt.xlim(tmin - 1.0, tmax + 1.0)
plt.ylim(e_min - escale * abs(e_min), e_max + escale * abs(e_max))
plt.legend(['negative potential', 'kinetic', "hamiltonian"], fontsize = 8)


# In[ ]:


plt.tight_layout()
fig = plt.gcf()
fig.savefig(f'n{tStep}_b{basis_timestep}.jpg', bbox_inches = 'tight')
plt.close('all')
# plt.show()
if tStep > 1e6:
    np.savez(f'trajs_n{tStep}_b{basis_timestep}_iter{XmaxIter}.npz',
        orbit = orbit, times = run_times
    )


logger.info('finish')

