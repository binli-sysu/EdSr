#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import matplotlib.pyplot as plt
from matplotlib import font_manager
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"
import numpy as np
from twoBodies import TwoBody
from data import *

fontsize = font_manager.FontProperties(size = 15)
tick_fontsize = font_manager.FontProperties(size = 15)
title_fontsize = font_manager.FontProperties(size = 15.5)
legned_fontsize = font_manager.FontProperties(size = 5.5)


# In[ ]:


np.random.seed(2)
np.set_printoptions(threshold = np.inf, linewidth = np.inf)


# In[ ]:


fig = plt.figure(figsize=[10,4], dpi=200)
# ax1, ax2 = fig.subplots(1, 2)
# print(type(ax1))


# In[ ]:


# tStart     = 0.
# tStep      = 20000
# interval   = 5
# basis_timestep = 0.01
# XmaxIter   = 10
# VmaxIter   = 10

# label_interval = 5
# label_basis_timestep = basis_timestep / label_interval
# label_length = tStep * interval

# assert tStep % interval == 0

# # params of intial state
# init_state_params = dict(
#     nbodies = 2,
#     mass = 1.0, 
#     min_radius = 1.0, 
#     max_radius = 2.0, 
#     orbit_noise = 0.05
# )
# init_state = state_generator(**init_state_params)

# model = TwoBody()

# label, label_times = model.makelabel(init_state, label_basis_timestep, label_interval, label_length)

# orbit, run_times = model.TrajectoryWithVV(init_state, interval * basis_timestep, tStep)

# orbit, run_times = model.generateInParallel(init_state, basis_timestep, interval, length = tStep, XmaxIter = XmaxIter, VmaxIter = VmaxIter)

# orbit = edsr
with np.load('trajs_iter50.npz') as data:
    # label, label_times = data['label'], data['label_times']
    # control = data['control'], control_times = data['control_times']
    orbit, run_times = data['edsr'], data['edsr_times']

skip = 20
cut_off = None

print(orbit[:skip])
print(run_times[:skip])
if cut_off is not None:
    orbit, run_times = orbit[:cut_off:skip], run_times[:cut_off:skip]
else:
    orbit, run_times = orbit[::skip], run_times[::skip]

p1: np.ndarray = orbit[..., 0, 1:3]
p2: np.ndarray = orbit[..., 1, 1:3]

pax1, pay1 = np.split(p1, 2, axis = -1)
pax1, pay1 = pax1.squeeze(axis = 1), pay1.squeeze(axis = 1)
pax2, pay2 = np.split(p2, 2, axis = -1)
pax2, pay2 = pax2.squeeze(axis = 1), pay2.squeeze(axis = 1)

print(pax1.shape, pax2.shape, pay1.shape, pay2.shape)
x1, y1, x2, y2 = [], [], [], []

x_min = min(pax1.min(), pax2.min())
x_max = max(pax1.max(), pax2.max())
y_min = min(pay1.min(), pay2.min())
y_max = max(pay1.max(), pay2.max())

plt.subplot(1, 2, 1)
sub1_line1, = plt.plot(pax1, pay1, 'o', markersize = 3)
sub1_line2, = plt.plot(pax2, pay2, 'o', markersize = 3)

# sub1_line1, = plt.plot(x1, y1, 'o', animated = True, markersize = 7)
# sub1_line2, = plt.plot(x2, y2, 'o', animated = True, markersize = 7)
print(type(sub1_line1), type(sub1_line2))
print(orbit.shape)

qscale = 0.5

# coord initialization
plt.xlabel(r'$\mathbf{q_{x}}$', fontsize = fontsize.get_size()) ; plt.ylabel(r'$\mathbf{q_{y}}$', fontsize = fontsize.get_size())
plt.axis()
plt.xlim(x_min - qscale * abs(x_min), x_max + qscale * abs(x_max))
plt.ylim(y_min - qscale * abs(y_min), y_max + qscale * abs(y_max))
plt.title('Trajectories', fontsize = title_fontsize.get_size(), fontweight = 'bold')
plt.legend([sub1_line1, sub1_line2], ['body 1 path','body 2 path'],fontsize = 8)


# In[ ]:


potential = potential_energy(orbit)
kinetic = kinetic_energy(orbit)
total = total_energy(orbit)

times = run_times
print(potential.shape, kinetic.shape, total.shape, times.shape)
pe, ke, tote, t = [], [], [], []

plt.subplot(1, 2, 2)
pe_line, = plt.plot(times, potential, 'o', markersize = 2.5)
ke_line, = plt.plot(times, kinetic, 'o', markersize = 2.5)
tote_line, = plt.plot(times, total, 'o', markersize = 2.5)
# pe_line, = plt.plot(t, pe, 'o', animated = True, markersize = 7)
# ke_line, = plt.plot(t, ke, 'o', animated = True, markersize = 7)
# tote_line, = plt.plot(t, tote, 'o', animated = True, markersize = 7)
tmin, tmax = times.min(), times.max()

e_min = min(potential.min(), kinetic.min(), total.min(), 0)
e_max = max(potential.max(), kinetic.max(), total.max())

print(f'min energy: {e_min}\nmax energy: {e_max}')
print(type(pe_line), type(ke_line), type(tote_line))
print(pax1.shape, pay1.shape, pax2.shape, pay2.shape, potential.shape, kinetic.shape, total.shape, times.shape)

frames = np.stack([pax1, pay1, pax2, pay2, potential, kinetic, total, times], axis = 1)
timesteps = 1
frames = frames[::timesteps]

escale = 0.3

# energy initialization
plt.axis()
plt.title("Energy", fontsize = title_fontsize.get_size(), fontweight = 'bold')
plt.xlabel('time', fontsize = fontsize.get_size(), fontweight = 'bold')
plt.xscale('symlog')
plt.xlim(tmin - 1.0, tmax + 1.0)
plt.ylim(e_min - escale * abs(e_min), e_max + escale * abs(e_max))
plt.legend([pe_line, ke_line, tote_line], ['potential', 'kinetic', 'total'], fontsize = 8)


# In[ ]:


plt.tight_layout()
fig = plt.gcf()
fig.savefig('twobody_longtime.jpg')
plt.close('all')

print('finish')