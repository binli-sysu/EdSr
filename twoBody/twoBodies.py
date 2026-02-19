from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial

import logging
import numpy as np
from numpy import ndarray

from einops import rearrange, repeat
# from torch import Tensor, einsum
from tqdm import tqdm, trange

from data import *

# np.set_printoptions(threshold = np.inf, linewidth = np.inf)

# G = 6.67e-2 # cm^3*g/s^2
G = 1.0

logging.basicConfig(
    level = logging.INFO,
    # format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    format='%(asctime)s - %(filename)s - %(levelname)s: %(message)s'
)

class TwoBody(object):

    def __init__(
        self,
    ) -> None:
        pass

    def gradient(self, coord: ndarray, mass: ndarray):
        if len(mass.shape) == 2:
            matrix = np.einsum("ij,kl->ik", mass, mass)
        elif len(mass.shape) == 1:
            matrix = np.einsum("i,j->ij", mass, mass)
        vec = rearrange(coord, "n d -> 1 n d") - rearrange(coord, "n d -> n 1 d")
        dsq = (vec ** 2).sum(axis = -1, keepdims = True)
        inv_dsq = np.where(dsq == 0, 0., 1 / dsq)
        force_matrix = matrix * inv_dsq * vec * (inv_dsq ** 0.5)
        force = force_matrix.sum(axis = 1)
        return force

    # def gradient(self, xn, mass):
        
    #     m = torch.from_numpy(mass).unsqueeze(dim = -1)
    #     q = torch.tensor(xn, requires_grad = True)

    #     potential: Tensor = self.potential(q, m)
    #     grad = torch.autograd.grad(potential, q)[0]
        
    #     return -grad.detach().numpy()

    # def potential(self, q: Tensor, mass: Tensor):
    #     natoms = q.shape[0]

    #     row, col = np.triu_indices(natoms, 1)

    #     r_ij = (q[row] - q[col]).square().sum(dim = -1).sqrt()
    #     Uenergy = (mass[col] * mass[row] * G / r_ij).sum()

    #     return -Uenergy

    # def gradient(self, xn, mass):
    #     return self.get_accelerations(xn, mass)

    def TrajectoryWithVV(self, init_state: ndarray, dt: float, length: int):
        logging.info("generate data with small step in parallel")
        trajs: ndarray = np.zeros((length, *init_state.shape))
        times: ndarray = np.arange(0, length) * dt

        trajs[0] = init_state
        acc = None

        for idx in range(1, length):
            
            next_state, acc = self.velocityVerletIntegration(trajs[idx - 1], dt, acc)

            trajs[idx] = next_state

        return trajs, times
    
    def velocityVerletIntegration(self, state, Dt, acc):

        x, v = state[:, 1:3], state[:, 3:5]
        mass = state[:, 0:1]
        massinv = 1. / mass

        if acc is None:
            acc = self.gradient(x, mass) * massinv

        nextx = x + v * Dt + 0.5 * Dt * Dt * acc 

        new_acc = self.gradient(nextx, mass) * massinv
        nextv = v + (acc + new_acc) * Dt * 0.5 

        nextState = np.concatenate([mass, nextx, nextv], axis = -1)

        return nextState, new_acc
    
    def computeEdSr_series(self, state, Dt, XmaxIter, VmaxIter, xi: None = None) -> ndarray:

        x, v = state[:, 1:3], state[:, 3:5]
        mass = state[:, 0:1]

        massinv = 1. / mass
        xxn = x.copy() if xi is None else xi[:, 1:3]
        xvn = x.copy() if xi is None else xi[:, 1:3]

        Dtsq = Dt * Dt

        for n in range(XmaxIter, 0, -1):
            xcoeff = 2.0 * n
            
            # * compute displacement
            # acc = np.einsum("ij,i->ij", self.gradient(xxn, mass), massinv)
            xacc = self.gradient(xxn, mass) * massinv
            # if n == 1:
            #     xvn = xxn
            dxx = v * Dt + xacc * Dtsq / xcoeff
            xxn = x + dxx / (xcoeff - 1)

        for n in range(VmaxIter, 0, -1):
            vcoeff = 2.0 * n
            # * compute velocity
            vacc = self.gradient(xvn, mass) * massinv
            dxv = vacc * Dt / (vcoeff - 1)
            xvn = (x + (v + dxv) * Dt / (vcoeff - 2)) if n > 1 else (v + dxv)
        # acc = self.gradient(xvn, mass) * massinv
        # new_acc = self.gradient(xxn, mass) * massinv
        # xvn = v + (acc + new_acc) * Dt * 0.5 

        # mass = rearrange(mass, 'b -> b 1')
        
        nextState = np.concatenate([mass, xxn, xvn], axis = -1)

        return nextState
    
    def computeEdSr_parallel(self, state, Dt, maxIter, xi: None = None) -> ndarray:

        mass = state[:, 0:1]

        # attn parallel
        with ThreadPoolExecutor(max_workers = 2) as executor:
            xres = executor.submit(self._computeX, state, Dt, maxIter, xi)
            vres = executor.submit(self._computeV, state, Dt, maxIter, xi)

            xxn, xvn = xres.result(), vres.result()
        
        nextState = np.concatenate([mass, xxn, xvn], axis = -1)

        return nextState
    
    def _computeX(self, state, Dt, maxIter, xi = None):

        x, v = state[:, 1:3], state[:, 3:5]
        mass = state[:, 0:1]

        massinv = 1. / mass
        xxn = x.copy() if xi is None else xi[:, 1:3]


        Dtsq = Dt * Dt

        for n in range(maxIter, 0, -1):
            xcoeff = 2.0 * n

            acc = self.gradient(xxn, mass) * massinv
            dxx = v * Dt + acc * Dtsq / xcoeff
            xxn = x + dxx / (xcoeff - 1)
        
        return xxn
    
    def _computeV(self, state, Dt, maxIter, xi = None):

        x, v = state[:, 1:3], state[:, 3:5]
        mass = state[:, 0:1]

        massinv = 1. / mass
        xvn = x.copy() if xi is None else xi[:, 1:3]

        for n in range(maxIter, 0, -1):
            vcoeff = 2.0 * n

            acc = self.gradient(xvn, mass) * massinv
            dxv = acc * Dt / (vcoeff - 1)
            xvn = (x + (v + dxv) * Dt / (vcoeff - 2)) if n > 1 else (v + dxv)

        return xvn
    
    def generateInSeries(self, init_state: ndarray, basis_timestep: float, interval: int, length: int, XmaxIter: int, VmaxIter: int):

        trajs: ndarray = np.zeros((length, *init_state.shape))

        trajs[0] = init_state

        for idx in range(length - 1):
            
            next_state = self.computeEdSr_series(trajs[idx], basis_timestep * interval, XmaxIter, VmaxIter)
            # next_state = self.computeEdSr_parallel(trajs[idx], basis_timestep * interval, maxIter)

            trajs[idx + 1] = next_state

        return trajs

    def generateInParallel(self, init_state: ndarray, basis_timestep: float, interval: int, length: int, XmaxIter: int, VmaxIter: int):
        logging.info("generate data with edsr in parallel")
        init_states: ndarray = np.zeros((interval, *init_state.shape))
        times: ndarray = np.arange(0, interval * length) * basis_timestep

        init_states[0] = init_state

        for idx in range(interval - 1):
            
            next_state = self.computeEdSr_series(init_states[idx], basis_timestep, XmaxIter, VmaxIter)

            init_states[idx + 1] = next_state


        with ProcessPoolExecutor(max_workers = interval) as executor:

            exefunc = partial(self.generateInSeries, basis_timestep = basis_timestep, interval = interval, length = length, XmaxIter = XmaxIter, VmaxIter = VmaxIter)

            futures = list(tqdm(executor.map(exefunc, init_states), total = interval, disable = True))

        trajs = np.stack(futures, axis = 0) # shape: [interval, length, initial state shape]

        trajs = rearrange(trajs, "interval length ... -> (length interval) ...")

        return trajs, times
    
    def makelabel(self, init_state: ndarray, basis_timestep: float, interval: int, length: int):
        
        trajs, times = self.TrajectoryWithVV(init_state, basis_timestep, interval * length)

        return trajs[::interval], times[::interval]
        

def RMSE(predict: ndarray, label: ndarray) -> ndarray:
    error_each_atom = np.sqrt(np.sum(np.square(predict - label), axis = -1))
    tot_error = np.mean(error_each_atom, axis = -1)
    return tot_error

def MAE(predict: ndarray, label: ndarray) -> ndarray:
    return np.mean(np.abs(predict - label), axis = (-1, -2))


if __name__ == "__main__":

    # import matplotlib.pyplot as plt
    # from matplotlib import font_manager
    np.seterr(divide = 'ignore')
    tStart     = 0.
    tStep      = 20_000_000
    interval   = 5
    basis_timestep = 0.01
    XmaxIter    = 50
    VmaxIter   = 50

    label_interval = 5
    label_basis_timestep = basis_timestep / label_interval
    label_length = tStep * interval

    assert tStep % interval == 0
    
    # params of intial state
    init_state_params = dict(
        nbodies = 2,
        mass = 1.0, 
        min_radius = 1.0, 
        max_radius = 2.0, 
        orbit_noise = 0.05
    )
    init_state = state_generator(**init_state_params)
    params = dict(
        tStart     = tStart,
        tStep      = tStep,
        interval   = interval,
        basis_timestep = basis_timestep,
        XmaxIter    = XmaxIter,
        VmaxIter   = VmaxIter,
        label_interval = label_interval,
        label_basis_timestep = label_basis_timestep,
        label_length = label_length
    )
    logging.info(f'{params.items()}')

    model = TwoBody()
    with ProcessPoolExecutor(max_workers = 3) as executor:
        labelres = executor.submit(model.makelabel, init_state, label_basis_timestep, label_interval, label_length)
        control_res = executor.submit(model.TrajectoryWithVV, init_state, basis_timestep, interval * tStep)
        edsr_res = executor.submit(model.generateInParallel, init_state, basis_timestep, interval, length = tStep, XmaxIter = XmaxIter, VmaxIter = VmaxIter)

        label, label_times = labelres.result()
        control, control_times = control_res.result()
        edsr, edsr_times = edsr_res.result()
    
    # label, label_times = model.makelabel(init_state, label_basis_timestep, label_interval, label_length)
    # logging.info("generate data with big step")
    # control, control_times = model.TrajectoryWithVV(init_state, basis_timestep, interval * tStep)
    
    # edsr, edsr_times = model.generateInParallel(init_state, basis_timestep, interval, length = tStep, XmaxIter = XmaxIter, VmaxIter = VmaxIter)

    logging.info("saving data")
    np.savez(f'trajs_iter{XmaxIter}.npz',
        label = label, labeltimes = label_times, 
        control = control, control_times = control_times,
        edsr = edsr, edsr_times = edsr_times
    )
