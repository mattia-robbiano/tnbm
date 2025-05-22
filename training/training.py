import sys
import copy
import pickle
from typing import Callable, Any
import numpy as np
import jax
import quimb.tensor as qtn

sys.path.append('../')
from functions.dataset_utils import bars_and_stripes, hypertn_from_data
from functions.loss import mmd_loss, nll_loss
import os
from datetime import datetime

jax.config.update("jax_enable_x64", True)

log_dir = f'log/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
os.makedirs(log_dir, exist_ok=True)                         # creating log folder

"""
Initialize the state to be optimized:
- psi: state to be optimized, inizialized as MPS with random state in with complex dtype
- chi: bond dimension of the MPS
- num_sites: number of sites in the MPS
"""
num_sites = 9

"""
Create MPO representation of the POVM.
p0 and p1 are the POVM projectors over computational base (onehot encoded).
Each node of the MPO is a tensor with 3 indices:
Indices o{} are used to label the output of the tensor (in this case 0 and 1 for each qubit)
k{} and b{} will be contracted with psi and psi.H 
"""
p0 = np.array([[1,0],[0,0]]) 
p1 = np.array([[0,0],[0,1]])
comp_basis_povm = np.array([p0, p1])
tensors = []
for i in range(num_sites):
    t = qtn.Tensor(comp_basis_povm,inds=(f'o{i}', f'k{i}', f'b{i}'), tags = [f'I{i}', 'POVM'])
    tensors.append(t)
povm_tn = qtn.TensorNetwork(tensors)
povm_tn = povm_tn / povm_tn.norm()

"""
Building the kernel matrix product operator (MPO) for the MMD loss function.
Each node of the MPO is a rank-2 tensor (matrix kernel_mat for gaussian kernel)
Index o{} and op{} is used to label the output of the tensor (in this case 0 and 1 for each qubit)
"""
sigma = 0.55
fact = np.exp(-1. / (2 * sigma**2))
kernel_mat = np.array([[1, fact],[fact, 1]])
kernel_mpo = qtn.MPO_product_operator([kernel_mat]*num_sites, 
                                      upper_ind_id = 'o{}', 
                                      lower_ind_id = 'op{}', 
                                      tags = ['KERNEL'])
kernel_mpo = kernel_mpo / kernel_mpo.norm()

"""
Create the dataset for the training, in form of a list of bitstrings. This dataset is then converted into a hyper tensor network by hypertn_from_data function, where each value of the hyperindex labels a different datapoint.
"""
bas = bars_and_stripes(int(np.sqrt(num_sites)), shuffle=True)
num_samples = bas.shape[0]
htn_freq = hypertn_from_data(data=bas)/num_samples


"""
Now we optimize the MPS to minimize the two objective functions, saving loss values, gradients and optimited state to log

- optimize() wrapper function is defined
- mmd optimization
- nll optimization
"""

def optimize(psi, loss_fn : Callable[...,float], loss_constants: dict, loss_kwargs: dict, iterations: int):
    """
    Optimize a tensor network state using a specified loss function and quimb.tensors.TNOptimizer parameters.
    Parameters:
        loss_fn (Callable[..., float]): The loss function to be minimized.
        loss_constants (dict): Dictionary of constant arguments to be passed to the loss function.
        loss_kwargs (dict): Dictionary of keyword arguments to be passed to the loss function.
        iterations (int): Number of optimization steps to perform.
        callback (Callable[..., float]): Callback function to be called during optimization (e.g., for logging or early stopping).
    Returns:
        Optimized tensor network state after the specified number of iterations.
    """
    tnopt = qtn.TNOptimizer(
                        tn = psi,
                        loss_fn = loss_fn,
                        loss_constants= loss_constants,
                        loss_kwargs= loss_kwargs,
                        norm_fn=lambda x: x / x.norm(),
                        autodiff_backend='jax',
                        jit_fn=False,
                        optimizer='adam',
                        )
    psi_opt = tnopt.optimize(iterations)
    return psi_opt, tnopt.losses


""" NLL optimization"""
for chi in [2, 8, 16, 32]:
    psi = qtn.MPS_rand_state(num_sites, chi, tags = ['PSI'])
    loss_constants={"htn_data":htn_freq}
    loss_kwargs={'contraction_method':'auto-hq'}
    psi_opt, losses = optimize(psi, nll_loss, loss_constants, loss_kwargs, 1000)

    with open(os.path.join(log_dir, f'psi_opt_nll_chi{chi}.pkl'), 'wb') as f: pickle.dump(psi_opt, f)
    with open(os.path.join(log_dir, f'losses_nll_chi{chi}.pkl'), 'wb') as f:  pickle.dump(losses, f)


""" MMD optimization"""
for chi in [2, 8, 16, 32]:
    psi = qtn.MPS_rand_state(num_sites, chi, tags = ['PSI'])
    loss_constants = {"povm_tn":povm_tn, "kernel_mpo":kernel_mpo, "htn_data":htn_freq}
    loss_kwargs={'contraction_method':'auto-hq'}
    psi_opt, losses = optimize(psi, mmd_loss, loss_constants, loss_kwargs, 1000)

    with open(os.path.join(log_dir, 'psi_opt_mmd_chi{chi}.pkl'), 'wb') as f:  pickle.dump(psi_opt, f)
    with open(os.path.join(log_dir, 'losses_mmd_chi{chi}.pkl'), 'wb') as f:   pickle.dump(losses, f)
