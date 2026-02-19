from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import numpy as np
import logging

from numpy import ndarray
from tqdm import tqdm

from einops import rearrange, repeat

# logging.basicConfig(
#     level = logging.INFO,
#     # format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
#     format='%(asctime)s - %(filename)s - %(levelname)s: %(message)s'
# )

class IdealSpring(object):

    def __init__(
        self,
        k       : float = 1.0,
        hamilton: float = 1.0,
    ) -> None:
        self.k = k
        self.hamilton = hamilton

    def gradient(self, x):
        return -x
    

    def computeEdSr(self, state, Dt, maxIter, split = 8):

        mass, x, v = state   

        massinv = 1. / mass 

        xn = np.array(x)
        vn = np.array(x)

        Dtsq = Dt * Dt

        # * compute displacement 
        for n in range(maxIter, 0, -1):
            xcoeff = 2.0 * n
            xacc = self.gradient(xn) * massinv
            dx = v * Dt + xacc * Dtsq / xcoeff 
            xn = x + dx / (xcoeff - 1)

            # * compute velocity
            vcoeff = 2.0 * n
            vacc = self.gradient(vn) * massinv
            dv = vacc * Dt / (vcoeff - 1)
            vn = (x + (v + dv) * Dt / (vcoeff - 2)) if n > 1 else (v + dv)

        nextState = np.concatenate([mass, xn, vn], axis = -1)
        return nextState
    
    def TrajectoryWithVV(self, init_state: ndarray, dt: float, length: int):

        trajs: ndarray = np.zeros((length, *init_state.shape))
        times: ndarray = np.arange(0, length) * dt

        trajs[0] = init_state
        acc = None

        for idx in tqdm(range(1, length), desc = "vv generation: "):
            
            next_state, acc = self.velocityVerletIntegration(trajs[idx - 1], dt, acc)

            trajs[idx] = next_state

        return trajs, times
    
    def velocityVerletIntegration(self, state, Dt, acc):

        x, v = state[1:2], state[2:]
        mass = state[0:1]
        massinv = 1. / mass

        if acc is None:
            acc = self.gradient(x) * massinv

        nextx = x + v * Dt + 0.5 * Dt * Dt * acc 

        new_acc = self.gradient(nextx) * massinv
        nextv = v + (acc + new_acc) * Dt * 0.5 

        nextState = np.concatenate([mass, nextx, nextv], axis = -1)

        return nextState, new_acc
    
    # attn the first experiment, get f(x + n*dx) by f(x + (n-1)*dx)
    
    def computeEdSr_series(self, state, Dt, XmaxIter, VmaxIter, xi: None = None) -> ndarray:

        x, v = state[1:2], state[2:]
        mass = state[0:1]

        massinv = 1. / mass
        xxn = x.copy() if xi is None else xi[1:2]
        xvn = x.copy() if xi is None else xi[1:2]

        Dtsq = Dt * Dt

        for n in range(XmaxIter, 0, -1):
            xcoeff = 2.0 * n
            
            # * compute displacement
            # acc = np.einsum("ij,i->ij", self.gradient(xxn, mass), massinv)
            xacc = self.gradient(xxn) * massinv
            # if n == 3:
            #     xvn = xxn.copy()
            dxx = v * Dt + xacc * Dtsq / xcoeff
            xxn = x + dxx / (xcoeff - 1)

        # xvn = xxn.copy()

        for n in range(VmaxIter, 0, -1):
            vcoeff = 2.0 * n
            # * compute velocity
            vacc = self.gradient(xvn) * massinv
            dxv = vacc * Dt / (vcoeff - 1)
            if n >= 2:
                xvn = (x + (v + dxv) * Dt / (vcoeff - 2)) 
            else:
                xvn = (v + dxv)

        # acc = self.gradient(xvn) * massinv
        # new_acc = self.gradient(xxn) * massinv
        # xvn = v + (acc + new_acc) * Dt * 0.5

        # mass = rearrange(mass, 'b -> b 1')
        
        nextState = np.concatenate([mass, xxn, xvn], axis = -1)

        return nextState
    
    def generateInParallel(self, init_state: ndarray, basis_timestep: float, interval: int, length: int, XmaxIter: int, VmaxIter: int):
        
        times: ndarray = np.arange(0, interval * length) * basis_timestep
        
        if interval == 1:
            trajs = self.generateInSeries(init_state, basis_timestep = basis_timestep, interval = interval, length = length, XmaxIter = XmaxIter, VmaxIter = VmaxIter, process_idx = 0)
            return trajs, times
        

        init_states: ndarray = np.zeros((interval, *init_state.shape))

        init_states[0] = init_state

        for idx in range(interval - 1):
            
            next_state = self.computeEdSr_series(init_states[idx], basis_timestep, XmaxIter, VmaxIter)

            init_states[idx + 1] = next_state
        
        with ProcessPoolExecutor(max_workers = interval) as executor:

            exefunc = partial(self.generateInSeries, basis_timestep = basis_timestep, interval = interval, length = length, XmaxIter = XmaxIter, VmaxIter = VmaxIter)
            futures = [executor.submit(exefunc, init_states[process_idx], process_idx = process_idx) for process_idx in range(interval)]
        
            # futures = list(tqdm(executor.map(exefunc, init_states), total = interval, disable = True))
            futures = list(map(lambda x: x.result(), futures))
        
        trajs = np.stack(futures, axis = 0) # shape: [interval, length, initial state shape]

        trajs = rearrange(trajs, "interval length ... -> (length interval) ...")
            
        return trajs, times
    
    def generateInSeries(self, init_state: ndarray, basis_timestep: float, interval: int, length: int, XmaxIter: int, VmaxIter: int, process_idx: int = 0):

        trajs: ndarray = np.zeros((length, *init_state.shape))

        trajs[0] = init_state

        breakpoint = 20

        for idx in range(length - 1):

            if process_idx == 0:
                if idx == 2 ** breakpoint:
                    print(f"Step {idx} has done.")
                    breakpoint += 1
            
            next_state = self.computeEdSr_series(trajs[idx], basis_timestep * interval, XmaxIter, VmaxIter)
            # next_state = self.computeEdSr_parallel(trajs[idx], basis_timestep * interval, maxIter)

            trajs[idx + 1] = next_state

        return trajs
    
    def makelabel(self, tStart: float, mass: float, basis_timestep: float, interval: int, length: int):
        
        times = tStart + np.arange(interval * length) * basis_timestep
        labelx   = np.sqrt(2 * self.hamilton) * np.sin(times)
        labelv   = np.sqrt(2 * self.hamilton) * np.cos(times)
        m = np.ones_like(labelx) * mass

        trajs = np.stack([m, labelx, labelv], axis = -1)
        init_state = trajs[0].copy()

        return init_state, trajs[::interval], times[::interval]

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    tStart     = 0.
    tStep      = 1000
    interval   = 10
    basis_timestep = 0.05
    XmaxIter    = 50
    VmaxIter   = 50

    label_interval = 5
    label_basis_timestep = basis_timestep / label_interval
    label_length = tStep * interval

    
    # print(times)

    mass     = 1.0
    hamilton = 1.0
    k        = 1.0

    model = IdealSpring(k = k, hamilton = hamilton)
    
    init_state, label, label_times = model.makelabel(tStart, mass, label_basis_timestep, label_interval, label_length)

    control, control_times = model.TrajectoryWithVV(init_state, basis_timestep, interval * tStep)

    edsr, edsr_times = model.generateInParallel(init_state, basis_timestep, interval, length = tStep, XmaxIter = XmaxIter, VmaxIter = VmaxIter)

    print(label_times)
    print(control_times)
    print(edsr_times)
    print(label.shape, edsr.shape, label_times.shape, edsr_times.shape)
    edsr_e = np.mean(abs(label - edsr), axis = (-2, -1))
    control_e = np.mean(abs(label - control), axis = (-2, -1))
    for ee, ce in zip(edsr_e, control_e):
        print(ee, ce)
