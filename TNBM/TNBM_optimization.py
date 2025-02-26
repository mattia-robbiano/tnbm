import quimb
import quimb.tensor as qtn
import numpy as np
import matplotlib.pyplot as plt

from functions import *

L = 9
D = 8
sigma = 0.09

# create a random MPS as our initial target to optimize
psi = qtn.MPS_rand_state(L, bond_dim=D)
Ommd = Ommd(L, sigma)
dataset = get_bars_and_stripes(3)

def loss_fn(psi,dataset,Ommd):
    loss = 0
    for data in dataset:
        y = qtn.MPS_computational_state(data)
        loss += MMD(psi, y, Ommd, sigma, L, D)
    loss = loss / len(dataset)
    return loss

tnopt = qtn.TNOptimizer(
    # the tensor network we want to optimize
    psi,
    # the functions specfying the loss and normalization
    loss_fn=loss_fn,
    #norm_fn=norm_fn,
    # we specify constants so that the arguments can be converted
    # to the  desired autodiff backend automatically
    loss_constants={"dataset": dataset, "Ommd": Ommd},
    # the underlying algorithm to use for the optimization
    # 'l-bfgs-b' is the default and often good for fast initial progress
    optimizer="adam",
    # which gradient computation backend to use
    autodiff_backend="numpy",
)
tnopt

psi_opt = tnopt.optimize(10)

tnopt.plot()