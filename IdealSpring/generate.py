#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# %matplotlib Qt6
import logging
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from idealSpring import IdealSpring

fontsize = font_manager.FontProperties(size = 15)
tick_fontsize = font_manager.FontProperties(size = 15)
title_fontsize = font_manager.FontProperties(size = 15.5)
legned_fontsize = font_manager.FontProperties(size = 5.5)

# logging.basicConfig(
#     level = logging.INFO,
#     # format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
#     format='%(asctime)s - %(filename)s - %(levelname)s: %(message)s'
# )

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
tStep      = 100_000_00
interval   = 20
basis_timestep = 0.5
XmaxIter    = 50
VmaxIter   = 50

label_interval = 5
label_basis_timestep = basis_timestep / label_interval
label_length = tStep * interval
# logging.info(times)

mass     = 1.0
hamilton = 1.0
k        = 1.0

def potential_energy(orbit):
    potential = 0.5 * k * orbit[:, 1] ** 2
    return potential

def kinetic_energy(orbit):
    kinetic = 0.5 * mass * orbit[:, 0] * orbit[:, 2] ** 2
    return kinetic

def hamiltonian(orbit):
    hamiltonian = 0.5 * k * orbit[:, 1] ** 2 + 0.5 * mass * orbit[:, 0] * orbit[:, 2] ** 2
    return hamiltonian

def lagrangian(orbit):
    lagrangian = 0.5 * mass * orbit[:, 0] * orbit[:, 2] ** 2 - 0.5 * k * orbit[:, 1] ** 2
    return lagrangian

model = IdealSpring(k = k, hamilton = hamilton)

init_state, label, label_times = model.makelabel(tStart, mass, label_basis_timestep, label_interval, label_length)

# control, control_times = model.TrajectoryWithVV(init_state, basis_timestep, interval * tStep)
logger.info("start to generate data in parallel")
orbit, run_times = model.generateInParallel(init_state, basis_timestep, interval, length = tStep, XmaxIter = XmaxIter, VmaxIter = VmaxIter)
logger.info('data have been generated')

# In[ ]:


x: np.ndarray = orbit[..., 1]
y: np.ndarray = orbit[..., 2]

x_min = x.min()
x_max = x.max()
y_min = y.min()
y_max = y.max()

plt.subplot(1, 2, 1)
plt.plot(x, y, 'o', markersize = 3)

sub1_line1, = plt.plot(x, y, 'o', animated = True, markersize = 4)
logger.info(type(sub1_line1))
logger.info(orbit.shape)

qscale = 0.5

# coord initialization
plt.xlabel(r'$\mathbf{q_{x}}$', fontsize = fontsize.get_size()) ; plt.ylabel(r'$\mathbf{q_{y}}$', fontsize = fontsize.get_size())
plt.axis()
plt.xlim(x_min - qscale * abs(x_min), x_max + qscale * abs(x_max))
plt.ylim(y_min - qscale * abs(y_min), y_max + qscale * abs(y_max))
plt.title('Trajectories', fontsize = title_fontsize.get_size(), fontweight = 'bold')
plt.legend([sub1_line1], ['hamiltonian path'],fontsize = 8)


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
plt.plot(times, hamilton, 'o', markersize = 1)
# plt.plot(times, lag, 'o', markersize = 2)
tmin, tmax = times.min(), times.max()

e_min = min(potential.min(), kinetic.min(), hamilton.min(), -1.0)
e_max = max(potential.max(), kinetic.max(), hamilton.max())

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
plt.legend(['negative potential', 'kinetic', 'hamiltonian'], fontsize = 8)


# In[ ]:

np.savez(f"n{tStep}_b{basis_timestep}_intv{interval}.npz", orbit = orbit, run_times = run_times)

plt.tight_layout()
fig = plt.gcf()
fig.savefig(f'n{tStep}_b{basis_timestep}.jpg', bbox_inches = 'tight')
plt.close('all')

logger.info('finish')

