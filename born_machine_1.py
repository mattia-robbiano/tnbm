#
# Definition of classes and functions to simulate a quantum circuit born machine. The simulation framework is
# pennylane + jax.
#
import jax
import jax.numpy as jnp
from functools import partial

jax.config.update("jax_enable_x64", True)

class QCBM:
#
# Defining QCBM object, storing circuit, loss and target distribution
#
# METHODS:
    
# __init__ ::

#    circ: quantum circuit
#    mmd : MMD object, loss function to be minimized
#    py  : target distribution

# mmd_loss ::

#    calculating px through circuit function (simulate quantum circuit and take shots for probability dist)
#    return mmd loss value and px. Keeping self fixed in jit, parameters will be optimized.
#    
    def __init__(self, circ, mmd, py):
        self.circ = circ
        self.mmd = mmd
        self.py = py

    @partial(jax.jit, static_argnums=0)
    def mmd_loss(self, params):
        px = self.circ(params)
        return self.mmd(px, self.py), px

class MMD:
#
# Define a MMD object, initialized once given the points on which we are evaluating exected value of kernel function
# and sigmas.
#
# METHODS:
#    
# __init__ ::
#    
#    scales: array of sigmas of the kernel functions we want to sum over
#    space:  array of points at which probability distributions will be evaluated when calculating expectation
#    value
#   
#    sq_dists: (space[:m None] - space[None, :]) the two 1d arrays are casted in a column and row matrices
#    subtracting, we are creating a matrix of pairwise distances (each squared by **2).
#
# k_expval ::
#    
#    returns expected value of K(x,y) over the distributions px, py, meaning we will extract x from px and
#    y from py
#
# __call__ :: returns the loss value, calculated with reduced expression, see google doc in readme for references    
#        
    def __init__(self, scales, space):
        gammas = 1 / (2 * (scales**2))
        sq_dists = jnp.abs(space[:, None] - space[None, :]) ** 2
        self.K = sum(jnp.exp(-gamma * sq_dists) for gamma in gammas) / len(scales)
        self.scales = scales

    def k_expval(self, px, py):
        return px @ self.K @ py

    def __call__(self, px, py):
        pxy = px - py
        return self.k_expval(pxy, pxy)
