#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# %matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import font_manager
from concurrent.futures import ThreadPoolExecutor
import matplotlib.image as mpimg
import numpy as np
from pprint import pprint

from einops import rearrange, repeat

from compute import *

# hype parameter
model_name = 'EdSr'
control_name = 'MD'
benchmark_name = 'BM'
""" Temp Press PotEng KinEng Enthalpy E_vdwl E_coul E_pair E_bond E_angle E_dihed E_long E_tail E_mol Ecouple Econserve TotEng Lx Ly Lz"""
energy_unit = "kcal $\\cdot$ mol$^{-1}$"
press_unit = "ATM"
temperature = "K"
distance_unit = "Angstrom"
time_unit = "fs"
thermo_style_unit = {
    'temp'     : f"Kelvin ({temperature})",    'Temp'     : f'Kelvin ({temperature})',
    'press'    : f'ATMosphere ({press_unit})', 'Press'    : f'ATMosphere ({press_unit})',
    "pe"       : f'energy ({energy_unit})',    'PotEng'   : f'energy ({energy_unit})',
    "ke"       : f'energy ({energy_unit})',    'KinEng'   : f'energy ({energy_unit})',
    "enthalpy" : f'energy ({energy_unit})',    'Enthalpy' : f'energy ({energy_unit})',
    "evdwl"    : f'energy ({energy_unit})',    'E_vdwl'   : f'energy ({energy_unit})',
    "ecoul"    : f'energy ({energy_unit})',    'E_coul'   : f'energy ({energy_unit})',
    "epair"    : f'energy ({energy_unit})',    'E_pair'   : f'energy ({energy_unit})',
    "ebond"    : f'energy ({energy_unit})',    'E_bond'   : f'energy ({energy_unit})',
    "eangle"   : f'energy ({energy_unit})',    'E_angle'  : f'energy ({energy_unit})',
    "edihed"   : f'energy ({energy_unit})',    'E_dihed'  : f'energy ({energy_unit})',
    "eimp"     : f'energy ({energy_unit})',
    "elong"    : f'energy ({energy_unit})',    'E_long'   : f'energy ({energy_unit})',
    "etail"    : f'energy ({energy_unit})',    'E_tail'   : f'energy ({energy_unit})',
    "emol"     : f'energy ({energy_unit})',    'E_mol'    : f'energy ({energy_unit})',
    "ecouple"  : f'energy ({energy_unit})',    'Ecouple'  : f'energy ({energy_unit})',
    "econserve": f'energy ({energy_unit})',    'Econserve': f'energy ({energy_unit})',
    "etotal"   : f'energy ({energy_unit})',    'TotEng'   : f'energy ({energy_unit})',
    'lx'       : f'length ({distance_unit})',  'Lx'       : f'length ({distance_unit})',
    'ly'       : f'length ({distance_unit})',  'Ly'       : f'length ({distance_unit})',
    'lz'       : f'length ({distance_unit})',  'Lz'       : f'length ({distance_unit})',
}

title_mapping = {
     'temp'     : r'Temperature',                                                'Temp'     : r'Temperature',
     'press'    : r'Pressure',                                                   'Press'    : r'Pressure',
     "pe"       : r'Potential energy',                                           'PotEng'   : r'potential energy',
     "ke"       : r'Kinetic energy',                                             'KinEng'   : r'Kinetic energy',
     "enthalpy" : r'Total energy (pe + ke)',                                     'Enthalpy' : r'Total energy (pe + ke)',
     "evdwl"    : r'Van der Waals pairwise energy',                              'E_vdwl'   : r'Van der Waals pairwise energy',
     "ecoul"    : r'Coulombic pairwise energy',                                  'E_coul'   : r'Coulombic pairwise energy',
     "epair"    : r'Pairwise energy',                                            'E_pair'   : r'Pairwise energy',
     "ebond"    : r'Bond energy',                                                'E_bond'   : r'Bond energy',
     "eangle"   : r'Angle energy',                                               'E_angle'  : r'Angle energy',
     "edihed"   : r'Dihedral energy',                                            'E_dihed'  : r'Dihedral energy',
     "eimp"     : r'Improper energy',
     "elong"    : r'Long-range kspace energy',                                   'E_long'   : r'Long-range kspace energy',
     "etail"    : r'Van der Waals energy long-range tail correction',            'E_tail'   : r'Van der Waals energy long-range tail correction',
     "emol"     : r'Intramolecular energy',                                      'E_mol'    : r'Intramolecular energy',
     "ecouple"  : r'Cumulative energy change due to thermo/baro statting fixes', 'Ecouple'  : r'Cumulative energy change due to thermo/baro statting fixes',
     "econserve": r'Etotal + ecouple',                                           'Econserve': r'Etotal + ecouple',
     "etotal"   : r'Total energy',                                               'TotEng'   : r'Total energy',
     'lx'       : r'Length of x-axis',                                           'Lx'       : r'Length of x-axis',
     'ly'       : r'Length of y-axis',                                           'Ly'       : r'Length of y-axis',
     'lz'       : r'Length of z-axis',                                           'Lz'       : r'Length of z-axis',
     'Rg'       : r'Radius of gyration',                                         'RG'       : r'Radius of gyration',
     'rmsd'     : r'RMSD',                                                       'RMSD'     : r'RMSD',
     'msd'      : r'MSD',                                                        'MSD'      : r'MSD',
}



# np.set_printoptions(threshold = np.inf, precision = 10, linewidth = np.inf, suppress = True)

fontsize = font_manager.FontProperties(size = 12)
tick_fontsize = font_manager.FontProperties(size = 10)
title_fontsize = font_manager.FontProperties(size = 12)
legned_fontsize = font_manager.FontProperties(size = 8)
item_fontsize = font_manager.FontProperties(size = 12)

tight_layout_arg = dict(
    top=0.906,
    bottom=0.096,
    left=0.077,
    right=0.991,
    hspace=0.222,
    wspace=0.167
)

row, col = 2, 4

fig = plt.figure(figsize=[10*col,20*row], dpi=256)

pic_idx = 0
subgraph_item = 0 + ord('a') - 1
item_pos = (-0.07, 1.1)


# ## $$\textit{ 0.1 t}$$

# In[ ]:


# intv100

maxIter   : int   = 20
intv      : int   = 10
basis     : float = 0.01
start     : int   = 1
end       : int   = 10000
group     : str   = None
indole_idx: int   = 1
prefix    : str   = 'DPD_nbor1.0'

benchmark_params = [
    f'data/benchmark_nve_basis{basis}_intv{intv}/' if prefix == '' else f'data/{prefix}_benchmark_nve_basis{basis}_intv{intv}/',
    f'frames{start}_{end}.npz', 
    f'GlobalVariable.npz',
]
control_params = [
    f'data/control_nve_basis{basis}_intv{intv}/' if prefix == '' else f'data/{prefix}_control_nve_basis{basis}_intv{intv}/',
    f'frames{start}_{end}.npz', 
    f'GlobalVariable.npz',
]
taylor_params = [
    f'data/{model_name}_nve_basis{basis}_intv{intv}_iter{maxIter}/' if prefix == '' else f'data/{prefix}_{model_name}_nve_basis{basis}_intv{intv}_iter{maxIter}/',
    f'frames{start}_{end}.npz', 
    f'GlobalVariable.npz',
]

# mass shape: (natoms,), id shape: (natoms,),  x shape: (ntrajs, natoms, 3), v shape: (ntrajs, natoms, 3)
with ThreadPoolExecutor(max_workers = 3) as executor:
    futures = [
        executor.submit(extraction, *benchmark_params, filter = group),
        executor.submit(extraction, *control_params, filter = group),
        executor.submit(extraction, *taylor_params , filter = group),
    ]

benchmark_x, benchmark_v, benchmark_delta_t, benchmark_mass, benchmark_atype, benchmark_id, benchmark_boundary, benchmark_heads, benchmark_ppties, benchmark_init_state, benchmark_last_state = futures[0].result()
control_x, control_v, control_delta_t, control_mass, control_atype, control_id, control_boundary, control_heads, control_ppties, control_init_state, control_last_state = futures[1].result()
taylor_x, taylor_v, taylor_delta_t, taylor_mass, taylor_atype, taylor_id, taylor_boundary, taylor_heads, taylor_ppties, taylor_init_state, taylor_last_state = futures[2].result()
pprint(benchmark_heads)
pprint(benchmark_ppties)

times = np.arange(end)*intv*basis


# In[ ]:


# kinetic energy
ppty = 'ke'

ke_bidx, = np.where(benchmark_heads == ppty); bke = benchmark_ppties[:, ke_bidx]
ke_cidx, = np.where(control_heads == ppty); cke = control_ppties[:, ke_cidx]
ke_tidx, = np.where(taylor_heads == ppty); tke = taylor_ppties[:, ke_tidx]

ke_diff_bc = np.fabs(cke - bke)
ke_diff_bt = np.fabs(tke - bke)

# ! Picture idx
pic_idx += 1
subgraph_item += 1

skip = 200
if skip > 1:
    gke_diff_bc = np.pad(np.mean(ke_diff_bc.reshape(-1, skip), axis = -1), pad_width = (1, 0), mode = 'constant', constant_values = ke_diff_bc[0])
    gke_diff_bt = np.pad(np.mean(ke_diff_bt.reshape(-1, skip), axis = -1), pad_width = (1, 0), mode = 'constant', constant_values = ke_diff_bt[0])
    group_times = np.pad(times[::skip] + (skip - 1)*intv*basis, pad_width = (1, 0), mode = 'constant', constant_values = 0)
else:
    gke_diff_bc = ke_diff_bc
    gke_diff_bt = ke_diff_bt
    group_times = times

ax = plt.subplot(row, col, 1)
plt.text(*item_pos, chr(subgraph_item), transform = ax.transAxes, fontsize = item_fontsize.get_size(), fontweight = 'bold', va = 'top', ha = 'left')

# ! title of column 
plt.text(0.5, 1.15, f"{title_mapping[ppty]}\n($\\mathrm{{kcal \\cdot  mol^{{-1}}}}$)", transform = ax.transAxes, fontsize = title_fontsize.get_size(), fontweight = 'bold', va = 'center', ha = 'center')

# ! title of row
plt.text(-0.2, 0.5, f"{basis*intv:.1f} fs", transform = ax.transAxes, fontsize = 15, fontweight = 'bold', va = 'center', ha = 'center')

# plt.plot(times[::skip], tke[::skip], label = f"{model_name}", linewidth = 2)
# plt.plot(times[::skip], cke[::skip], label = f"{control_name}", linewidth = 1)
# plt.plot(times[::skip], bke[::skip], label = f"benchmark", linewidth = 1)
plt.plot(group_times, gke_diff_bt, label = f"{model_name}", linewidth = 2)
plt.plot(group_times, gke_diff_bc, label = f"{control_name}", linewidth = 1)
plt.axhline(y = np.mean(ke_diff_bt), color = 'g', linestyle = '--', linewidth = 2, label = f'$\\mathrm{{MAE_{{{model_name}}}}}$')
plt.axhline(y = np.mean(ke_diff_bc), color = 'r', linestyle = '--', linewidth = 1, label = f'$\\mathrm{{MAE_{{{control_name}}}}}$')

# plt.xlabel(f't (fs)', fontproperties = fontsize); # plt.ylabel(r'error (kcal$\cdot \mathrm{mol^{-1}}$)', fontproperties = fontsize)
# plt.title(f"{title_mapping[ppty]} (kcal$\\cdot \\mathrm{{mol^{{-1}}}}$)", loc = 'center', fontproperties = title_fontsize)
plt.ticklabel_format(axis = 'x', style = 'scientific', scilimits = (0, 2), useMathText = True, useLocale = True)
plt.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0, 2), useMathText = True, useLocale = True)
plt.tick_params(axis = 'both', labelsize = tick_fontsize.get_size())
# exp_idx = int(np.floor(np.log10(data_mean)))
# plt.yscale('log'); plt.yticks([1e1, 1e2])
plt.yticks([0, 5, 10, 15])

plt.legend(fontsize = legned_fontsize.get_size(), loc = 'best', ncol = 1)


# In[ ]:


# Pairwise energy
ppty = 'epair'

epair_bidx, = np.where(benchmark_heads == ppty); bepair = np.squeeze(benchmark_ppties[:, epair_bidx], axis = -1)
epair_cidx, = np.where(control_heads == ppty); cepair = np.squeeze(control_ppties[:, epair_cidx], axis = -1)
epair_tidx, = np.where(taylor_heads == ppty); tepair = np.squeeze(taylor_ppties[:, epair_tidx], axis = -1)

print('benchmark\n\n', bepair)
print('\ncontrol\n\n', cepair)
print('\ntaylor\n\n', tepair)

epair_diff_bc = np.fabs(cepair - bepair)
epair_diff_bt = np.fabs(tepair - bepair)

# ! Picture idx
pic_idx += 1
subgraph_item += 1

skip = 200
if skip > 1:
    gepair_diff_bc = np.pad(np.mean(epair_diff_bc.reshape(-1, skip), axis = -1), pad_width = (1, 0), mode = 'constant', constant_values = epair_diff_bc[0])
    gepair_diff_bt = np.pad(np.mean(epair_diff_bt.reshape(-1, skip), axis = -1), pad_width = (1, 0), mode = 'constant', constant_values = epair_diff_bt[0])
    group_times = np.pad(times[::skip] + (skip - 1)*intv*basis, pad_width = (1, 0), mode = 'constant', constant_values = 0)
else:
    gepair_diff_bc = epair_diff_bc
    gepair_diff_bt = epair_diff_bt
    group_times = times

ax = plt.subplot(row, col, pic_idx)
plt.text(*item_pos, chr(subgraph_item), transform = ax.transAxes, fontsize = item_fontsize.get_size(), fontweight = 'bold', va = 'top', ha = 'left')

# ! title of column 
plt.text(0.5, 1.15, f"{title_mapping[ppty]}\n($\\mathrm{{kcal \\cdot  mol^{{-1}}}}$)", transform = ax.transAxes, fontsize = title_fontsize.get_size(), fontweight = 'bold', va = 'center', ha = 'center')

# plt.plot(times[::skip], tepair[::skip], label = f"{model_name}", linewidth = 2)
# plt.plot(times[::skip], cepair[::skip], label = f"{control_name}", linewidth = 1)
# plt.plot(times[::skip], bepair[::skip], label = f"benchmark", linewidth = 1)
plt.plot(group_times, gepair_diff_bt, label = f"{model_name}", linewidth = 2)
plt.plot(group_times, gepair_diff_bc, label = f"{control_name}", linewidth = 1)
plt.axhline(y = np.mean(epair_diff_bt), color = 'g', linestyle = '--', linewidth = 2, label = f'$\\mathrm{{MAE_{{{model_name}}}}}$')
plt.axhline(y = np.mean(epair_diff_bc), color = 'r', linestyle = '--', linewidth = 1, label = f'$\\mathrm{{MAE_{{{control_name}}}}}$')

# plt.xlabel(f't (fs)', fontproperties = fontsize); # plt.ylabel(r'error (kcal$\cdot \mathrm{mol^{-1}}$)', fontproperties = fontsize)
# plt.title(f"{title_mapping[ppty]} (kcal$\\cdot \\mathrm{{mol^{{-1}}}}$)", loc = 'center', fontsize = title_fontsize.get_size())
plt.ticklabel_format(axis = 'x', style = 'scientific', scilimits = (0, 2), useMathText = True, useLocale = True)
plt.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0, 2), useMathText = True, useLocale = True)
plt.tick_params(axis = 'both', labelsize = tick_fontsize.get_size())
# exp_idx = int(np.floor(np.log10(data_mean)))
# plt.yscale('log')
# plt.yticks([0, 30, 60])
# plt.ylim(top = 6.8)

plt.legend(fontsize = legned_fontsize.get_size(), loc = 'upper left', ncol = 2)


# In[ ]:


# Molecular energy
ppty = 'emol'

emol_bidx, = np.where(benchmark_heads == ppty); bemol = benchmark_ppties[:, emol_bidx]
emol_cidx, = np.where(control_heads == ppty); cemol = control_ppties[:, emol_cidx]
emol_tidx, = np.where(taylor_heads == ppty); temol = taylor_ppties[:, emol_tidx]

emol_diff_bc = np.fabs(cemol - bemol)
emol_diff_bt = np.fabs(temol - bemol)

# ! Picture idx
pic_idx += 1
subgraph_item += 1

skip = 200
if skip > 1:
    gemol_diff_bc = np.pad(np.mean(emol_diff_bc.reshape(-1, skip), axis = -1), pad_width = (1, 0), mode = 'constant', constant_values = emol_diff_bc[0])
    gemol_diff_bt = np.pad(np.mean(emol_diff_bt.reshape(-1, skip), axis = -1), pad_width = (1, 0), mode = 'constant', constant_values = emol_diff_bt[0])
    group_times = np.pad(times[::skip] + (skip - 1)*intv*basis, pad_width = (1, 0), mode = 'constant', constant_values = 0)
else:
    gemol_diff_bc = emol_diff_bc
    gemol_diff_bt = emol_diff_bt
    group_times = times

ax = plt.subplot(row, col, pic_idx)
plt.text(*item_pos, chr(subgraph_item), transform = ax.transAxes, fontsize = item_fontsize.get_size(), fontweight = 'bold', va = 'top', ha = 'left')

# ! title of column 
plt.text(0.5, 1.15, f"{title_mapping[ppty]}\n($\\mathrm{{kcal \\cdot  mol^{{-1}}}}$)", transform = ax.transAxes, fontsize = title_fontsize.get_size(), fontweight = 'bold', va = 'center', ha = 'center')

# plt.plot(times[::skip], temol[::skip], label = f"{model_name}", linewidth = 2)
# plt.plot(times[::skip], cemol[::skip], label = f"{control_name}", linewidth = 1)
# plt.plot(times[::skip], bemol[::skip], label = f"benchmark", linewidth = 1)
plt.plot(group_times, gemol_diff_bt, label = f"{model_name}", linewidth = 2)
plt.plot(group_times, gemol_diff_bc, label = f"{control_name}", linewidth = 1)
plt.axhline(y = np.mean(emol_diff_bt), color = 'g', linestyle = '--', linewidth = 2, label = f'$\\mathrm{{MAE_{{{model_name}}}}}$')
plt.axhline(y = np.mean(emol_diff_bc), color = 'r', linestyle = '--', linewidth = 1, label = f'$\\mathrm{{MAE_{{{control_name}}}}}$')

# plt.xlabel(f't (fs)', fontproperties = fontsize); # plt.ylabel(r'error (kcal$\cdot \mathrm{mol^{-1}}$)', fontproperties = fontsize)
# plt.title(f"{title_mapping[ppty]} (kcal$\\cdot \\mathrm{{mol^{{-1}}}}$)", loc = 'center', fontsize = title_fontsize.get_size())
plt.ticklabel_format(axis = 'x', style = 'scientific', scilimits = (0, 2), useMathText = True, useLocale = True)
# plt.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0, 2), useMathText = True, useLocale = True)
plt.tick_params(axis = 'both', labelsize = tick_fontsize.get_size())
# exp_idx = int(np.floor(np.log10(data_mean)))
# plt.yscale('log')
# plt.yticks([0, 5, 10, 15])
# plt.yscale('log'); plt.yticks([1e1, 4e1])
# plt.ylim(top = 52)

plt.legend(fontsize = legned_fontsize.get_size(), loc = 'best', ncol = 1)


# In[ ]:


# press
ppty = 'press'

press_bidx, = np.where(benchmark_heads == ppty); bpress = benchmark_ppties[:, press_bidx]
press_cidx, = np.where(control_heads == ppty); cpress = control_ppties[:, press_cidx]
press_tidx, = np.where(taylor_heads == ppty); tpress = taylor_ppties[:, press_tidx]

press_diff_bc = np.fabs(cpress - bpress)
press_diff_bt = np.fabs(tpress - bpress)

# ! Picture idx
pic_idx += 1
subgraph_item += 1

skip = 200
if skip > 1:
    gpress_diff_bc = np.pad(np.mean(press_diff_bc.reshape(-1, skip), axis = -1), pad_width = (1, 0), mode = 'constant', constant_values = press_diff_bc[0])
    gpress_diff_bt = np.pad(np.mean(press_diff_bt.reshape(-1, skip), axis = -1), pad_width = (1, 0), mode = 'constant', constant_values = press_diff_bt[0])
    group_times = np.pad(times[::skip] + (skip - 1)*intv*basis, pad_width = (1, 0), mode = 'constant', constant_values = 0)
else:
    gpress_diff_bc = press_diff_bc
    gpress_diff_bt = press_diff_bt
    group_times = times

ax = plt.subplot(row, col, pic_idx)
plt.text(*item_pos, chr(subgraph_item), transform = ax.transAxes, fontsize = item_fontsize.get_size(), fontweight = 'bold', va = 'top', ha = 'left')

# ! title of column 
plt.text(0.5, 1.15, f"{title_mapping[ppty]}\n($\\mathrm{{ATM}}$)", transform = ax.transAxes, fontsize = title_fontsize.get_size(), fontweight = 'bold', va = 'center', ha = 'center')

# plt.plot(times[::skip], tpress[::skip], label = f"{model_name}", linewidth = 2)
# plt.plot(times[::skip], cpress[::skip], label = f"{control_name}", linewidth = 1)
# plt.plot(times[::skip], bpress[::skip], label = f"benchmark", linewidth = 1)
plt.plot(group_times, gpress_diff_bt, label = f"{model_name}", linewidth = 2)
plt.plot(group_times, gpress_diff_bc, label = f"{control_name}", linewidth = 1)
plt.axhline(y = np.mean(press_diff_bt), color = 'g', linestyle = '--', linewidth = 2, label = f'$\\mathrm{{MAE_{{{model_name}}}}}$')
plt.axhline(y = np.mean(press_diff_bc), color = 'r', linestyle = '--', linewidth = 1, label = f'$\\mathrm{{MAE_{{{control_name}}}}}$')

# plt.xlabel(f't (fs)', fontproperties = fontsize); # plt.ylabel(r'error (kcal$\cdot \mathrm{mol^{-1}}$)', fontproperties = fontsize)
# plt.title(f"{title_mapping[ppty]} (ATM)", loc = 'center', fontsize = title_fontsize.get_size() - 1)
plt.ticklabel_format(axis = 'x', style = 'scientific', scilimits = (0, 2), useMathText = True, useLocale = True)
plt.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0, 2), useMathText = True, useLocale = True)
plt.tick_params(axis = 'both', labelsize = tick_fontsize.get_size())
# exp_idx = int(np.floor(np.log10(data_mean)))
# plt.yscale('log')
# plt.yticks([0, 10, 20, 30])

plt.legend(fontsize = legned_fontsize.get_size(), loc = 'best', ncol = 1)


# ## $$\textit{ 0.12\ t }$$

# In[ ]:


maxIter   : int   = 50
intv      : int   = 12
basis     : float = 0.01
start     : int   = 1
end       : int   = 10000
group     : str   = None
indole_idx: int   = 1
prefix    : str   = 'DPD_nbor1.0'

benchmark_params = [
    f'data/benchmark_nve_basis{basis}_intv{intv}/' if prefix == '' else f'data/{prefix}_benchmark_nve_basis{basis}_intv{intv}/',
    f'frames{start}_{end}.npz', 
    f'GlobalVariable.npz',
]
control_params = [
    f'data/control_nve_basis{basis}_intv{intv}/' if prefix == '' else f'data/{prefix}_control_nve_basis{basis}_intv{intv}/',
    f'frames{start}_{end}.npz', 
    f'GlobalVariable.npz',
]
taylor_params = [
    f'data/{model_name}_nve_basis{basis}_intv{intv}_iter{maxIter}/' if prefix == '' else f'data/{prefix}_{model_name}_nve_basis{basis}_intv{intv}_iter{maxIter}/',
    f'frames{start}_{end}.npz', 
    f'GlobalVariable.npz',
]

# mass shape: (natoms,), id shape: (natoms,),  x shape: (ntrajs, natoms, 3), v shape: (ntrajs, natoms, 3)
with ThreadPoolExecutor(max_workers = 3) as executor:
    futures = [
        executor.submit(extraction, *benchmark_params, filter = group),
        executor.submit(extraction, *control_params, filter = group),
        executor.submit(extraction, *taylor_params , filter = group),
    ]

benchmark_x, benchmark_v, benchmark_delta_t, benchmark_mass, benchmark_atype, benchmark_id, benchmark_boundary, benchmark_heads, benchmark_ppties, benchmark_init_state, benchmark_last_state = futures[0].result()
control_x, control_v, control_delta_t, control_mass, control_atype, control_id, control_boundary, control_heads, control_ppties, control_init_state, control_last_state = futures[1].result()
taylor_x, taylor_v, taylor_delta_t, taylor_mass, taylor_atype, taylor_id, taylor_boundary, taylor_heads, taylor_ppties, taylor_init_state, taylor_last_state = futures[2].result()

times = np.arange(end)*intv*basis


# In[ ]:


# kinetic energy
ppty = 'ke'

ke_bidx, = np.where(benchmark_heads == ppty); bke = benchmark_ppties[:, ke_bidx]
ke_cidx, = np.where(control_heads == ppty); cke = control_ppties[:, ke_cidx]
ke_tidx, = np.where(taylor_heads == ppty); tke = taylor_ppties[:, ke_tidx]

ke_diff_bc = np.fabs(cke - bke)
ke_diff_bt = np.fabs(tke - bke)

# ! Picture idx
pic_idx += 1
subgraph_item += 1

skip = 200
if skip > 1:
    gke_diff_bc = np.pad(np.mean(ke_diff_bc.reshape(-1, skip), axis = -1), pad_width = (1, 0), mode = 'constant', constant_values = ke_diff_bc[0])
    gke_diff_bt = np.pad(np.mean(ke_diff_bt.reshape(-1, skip), axis = -1), pad_width = (1, 0), mode = 'constant', constant_values = ke_diff_bt[0])
    group_times = np.pad(times[::skip] + (skip - 1)*intv*basis, pad_width = (1, 0), mode = 'constant', constant_values = 0)
else:
    gke_diff_bc = ke_diff_bc
    gke_diff_bt = ke_diff_bt
    group_times = times

ax = plt.subplot(row, col, pic_idx)
plt.text(*item_pos, chr(subgraph_item), transform = ax.transAxes, fontsize = item_fontsize.get_size(), fontweight = 'bold', va = 'top', ha = 'left')

# ! title of row
plt.text(-0.2, 0.5, f"{basis*intv:.1f} fs", transform = ax.transAxes, fontsize = 15, fontweight = 'bold', va = 'center', ha = 'center')

# plt.plot(times[::skip], tke[::skip], label = f"{model_name}", linewidth = 2)
# plt.plot(times[::skip], cke[::skip], label = f"{control_name}", linewidth = 1)
# plt.plot(times[::skip], bke[::skip], label = f"benchmark", linewidth = 1)
plt.plot(group_times, gke_diff_bt, label = f"{model_name}", linewidth = 2)
plt.plot(group_times, gke_diff_bc, label = f"{control_name}", linewidth = 1)
plt.axhline(y = np.mean(ke_diff_bt), color = 'g', linestyle = '--', linewidth = 2, label = f'$\\mathrm{{MAE_{{{model_name}}}}}$')
plt.axhline(y = np.mean(ke_diff_bc), color = 'r', linestyle = '--', linewidth = 1, label = f'$\\mathrm{{MAE_{{{control_name}}}}}$')

plt.xlabel(f't ($\\mathrm{{fs}}$)', fontproperties = fontsize, fontweight = 'bold'); # plt.ylabel(r'error (kcal$\cdot \mathrm{mol^{-1}}$)', fontproperties = fontsize)
# plt.title(f"{title_mapping[ppty]} (kcal$\\cdot \\mathrm{{mol^{{-1}}}}$)", loc = 'center', fontproperties = title_fontsize)
plt.ticklabel_format(axis = 'x', style = 'scientific', scilimits = (0, 2), useMathText = True, useLocale = True)
plt.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0, 2), useMathText = True, useLocale = True)
plt.tick_params(axis = 'both', labelsize = tick_fontsize.get_size())
# exp_idx = int(np.floor(np.log10(data_mean)))
# plt.yscale('log'); plt.yticks([1e1, 1e2])
plt.yticks([0, 5, 10, 15])

plt.legend(fontsize = legned_fontsize.get_size(), loc = 'best', ncol = 1)


# In[ ]:


# Pairwise energy
ppty = 'epair'

epair_bidx, = np.where(benchmark_heads == ppty); bepair = benchmark_ppties[:, epair_bidx]
epair_cidx, = np.where(control_heads == ppty); cepair = control_ppties[:, epair_cidx]
epair_tidx, = np.where(taylor_heads == ppty); tepair = taylor_ppties[:, epair_tidx]

print('benchmark\n\n', bepair)
print('\ncontrol\n\n', cepair)
print('\ntaylor\n\n', tepair)

epair_diff_bc = np.fabs(cepair - bepair)
epair_diff_bt = np.fabs(tepair - bepair)

# ! Picture idx
pic_idx += 1
subgraph_item += 1

skip = 200
if skip > 1:
    gepair_diff_bc = np.pad(np.mean(epair_diff_bc.reshape(-1, skip), axis = -1), pad_width = (1, 0), mode = 'constant', constant_values = epair_diff_bc[0])
    gepair_diff_bt = np.pad(np.mean(epair_diff_bt.reshape(-1, skip), axis = -1), pad_width = (1, 0), mode = 'constant', constant_values = epair_diff_bt[0])
    group_times = np.pad(times[::skip] + (skip - 1)*intv*basis, pad_width = (1, 0), mode = 'constant', constant_values = 0)
else:
    gepair_diff_bc = epair_diff_bc
    gepair_diff_bt = epair_diff_bt
    group_times = times

ax = plt.subplot(row, col, pic_idx)

# plt.plot(times[::skip], tepair[::skip], label = f"{model_name}", linewidth = 2)
# plt.plot(times[::skip], cepair[::skip], label = f"{control_name}", linewidth = 1)
# plt.plot(times[::skip], bepair[::skip], label = f"benchmark", linewidth = 1)
plt.plot(group_times, gepair_diff_bt, label = f"{model_name}", linewidth = 2)
plt.plot(group_times, gepair_diff_bc, label = f"{control_name}", linewidth = 1)
plt.axhline(y = np.mean(epair_diff_bt), color = 'g', linestyle = '--', linewidth = 2, label = f'$\\mathrm{{MAE_{{{model_name}}}}}$')
plt.axhline(y = np.mean(epair_diff_bc), color = 'r', linestyle = '--', linewidth = 1, label = f'$\\mathrm{{MAE_{{{control_name}}}}}$')

plt.xlabel(f't ($\\mathrm{{fs}}$)', fontproperties = fontsize, fontweight = 'bold'); # plt.ylabel(r'error (kcal$\cdot \mathrm{mol^{-1}}$)', fontproperties = fontsize)
# plt.title(f"{title_mapping[ppty]} (kcal$\\cdot \\mathrm{{mol^{{-1}}}}$)", loc = 'center', fontsize = title_fontsize.get_size())
plt.ticklabel_format(axis = 'x', style = 'scientific', scilimits = (0, 2), useMathText = True, useLocale = True)
plt.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0, 2), useMathText = True, useLocale = True)
plt.tick_params(axis = 'both', labelsize = tick_fontsize.get_size())
# exp_idx = int(np.floor(np.log10(data_mean)))
# plt.yscale('log')
# plt.yticks([0, 30, 60])
# plt.ylim(top = 6.8)

plt.legend(fontsize = legned_fontsize.get_size(), loc = 'upper left', ncol = 2)


# In[ ]:


# Molecular energy
ppty = 'emol'

emol_bidx, = np.where(benchmark_heads == ppty); bemol = benchmark_ppties[:, emol_bidx]
emol_cidx, = np.where(control_heads == ppty); cemol = control_ppties[:, emol_cidx]
emol_tidx, = np.where(taylor_heads == ppty); temol = taylor_ppties[:, emol_tidx]

emol_diff_bc = np.fabs(cemol - bemol)
emol_diff_bt = np.fabs(temol - bemol)

# ! Picture idx
pic_idx += 1
subgraph_item += 1

skip = 200
if skip > 1:
    gemol_diff_bc = np.pad(np.mean(emol_diff_bc.reshape(-1, skip), axis = -1), pad_width = (1, 0), mode = 'constant', constant_values = emol_diff_bc[0])
    gemol_diff_bt = np.pad(np.mean(emol_diff_bt.reshape(-1, skip), axis = -1), pad_width = (1, 0), mode = 'constant', constant_values = emol_diff_bt[0])
    group_times = np.pad(times[::skip] + (skip - 1)*intv*basis, pad_width = (1, 0), mode = 'constant', constant_values = 0)
else:
    gemol_diff_bc = emol_diff_bc
    gemol_diff_bt = emol_diff_bt
    group_times = times

ax = plt.subplot(row, col, pic_idx)

# plt.plot(times[::skip], temol[::skip], label = f"{model_name}", linewidth = 2)
# plt.plot(times[::skip], cemol[::skip], label = f"{control_name}", linewidth = 1)
# plt.plot(times[::skip], bemol[::skip], label = f"benchmark", linewidth = 1)
plt.plot(group_times, gemol_diff_bt, label = f"{model_name}", linewidth = 2)
plt.plot(group_times, gemol_diff_bc, label = f"{control_name}", linewidth = 1)
plt.axhline(y = np.mean(emol_diff_bt), color = 'g', linestyle = '--', linewidth = 2, label = f'$\\mathrm{{MAE_{{{model_name}}}}}$')
plt.axhline(y = np.mean(emol_diff_bc), color = 'r', linestyle = '--', linewidth = 1, label = f'$\\mathrm{{MAE_{{{control_name}}}}}$')

plt.xlabel(f't ($\\mathrm{{fs}}$)', fontproperties = fontsize, fontweight = 'bold'); # plt.ylabel(r'error (kcal$\cdot \mathrm{mol^{-1}}$)', fontproperties = fontsize)
# plt.title(f"{title_mapping[ppty]} (kcal$\\cdot \\mathrm{{mol^{{-1}}}}$)", loc = 'center', fontsize = title_fontsize.get_size())
plt.ticklabel_format(axis = 'x', style = 'scientific', scilimits = (0, 2), useMathText = True, useLocale = True)
# plt.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0, 2), useMathText = True, useLocale = True)
plt.tick_params(axis = 'both', labelsize = tick_fontsize.get_size())
# exp_idx = int(np.floor(np.log10(data_mean)))
# plt.yscale('log')
# plt.yticks([0, 5, 10, 15])
# plt.yscale('log'); plt.yticks([1e1, 4e1])
# plt.ylim(top = 52)

plt.legend(fontsize = legned_fontsize.get_size(), loc = 'best', ncol = 1)


# In[ ]:


# press
ppty = 'press'

press_bidx, = np.where(benchmark_heads == ppty); bpress = benchmark_ppties[:, press_bidx]
press_cidx, = np.where(control_heads == ppty); cpress = control_ppties[:, press_cidx]
press_tidx, = np.where(taylor_heads == ppty); tpress = taylor_ppties[:, press_tidx]

press_diff_bc = np.fabs(cpress - bpress)
press_diff_bt = np.fabs(tpress - bpress)

# ! Picture idx
pic_idx += 1
subgraph_item += 1

skip = 200
if skip > 1:
    gpress_diff_bc = np.pad(np.mean(press_diff_bc.reshape(-1, skip), axis = -1), pad_width = (1, 0), mode = 'constant', constant_values = press_diff_bc[0])
    gpress_diff_bt = np.pad(np.mean(press_diff_bt.reshape(-1, skip), axis = -1), pad_width = (1, 0), mode = 'constant', constant_values = press_diff_bt[0])
    group_times = np.pad(times[::skip] + (skip - 1)*intv*basis, pad_width = (1, 0), mode = 'constant', constant_values = 0)
else:
    gpress_diff_bc = press_diff_bc
    gpress_diff_bt = press_diff_bt
    group_times = times

ax = plt.subplot(row, col, pic_idx)

# plt.plot(times[::skip], tpress[::skip], label = f"{model_name}", linewidth = 2)
# plt.plot(times[::skip], cpress[::skip], label = f"{control_name}", linewidth = 1)
# plt.plot(times[::skip], bpress[::skip], label = f"benchmark", linewidth = 1)
plt.plot(group_times, gpress_diff_bt, label = f"{model_name}", linewidth = 2)
plt.plot(group_times, gpress_diff_bc, label = f"{control_name}", linewidth = 1)
plt.axhline(y = np.mean(press_diff_bt), color = 'g', linestyle = '--', linewidth = 2, label = f'$\\mathrm{{MAE_{{{model_name}}}}}$')
plt.axhline(y = np.mean(press_diff_bc), color = 'r', linestyle = '--', linewidth = 1, label = f'$\\mathrm{{MAE_{{{control_name}}}}}$')

plt.xlabel(f't ($\\mathrm{{fs}}$)', fontproperties = fontsize, fontweight = 'bold'); # plt.ylabel(r'error (kcal$\cdot \mathrm{mol^{-1}}$)', fontproperties = fontsize)
# plt.title(f"{title_mapping[ppty]} (ATM)", loc = 'center', fontsize = title_fontsize.get_size() - 1)
plt.ticklabel_format(axis = 'x', style = 'scientific', scilimits = (0, 2), useMathText = True, useLocale = True)
plt.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0, 2), useMathText = True, useLocale = True)
plt.tick_params(axis = 'both', labelsize = tick_fontsize.get_size())
# exp_idx = int(np.floor(np.log10(data_mean)))
# plt.yscale('log')
# plt.yticks([0, 10, 20, 30])

plt.legend(fontsize = legned_fontsize.get_size(), loc = 'best', ncol = 1)


# In[ ]:


fig.subplots_adjust()
plt.show()
# fig.savefig(f'ke_epair_emol_press_10_12.jpg')

