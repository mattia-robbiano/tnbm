import numpy as np
import math
import quimb as qu
import jax
import jax.numpy as jnp
import quimb as qu
import quimb.tensor as qtn


from  born_machine import QCBM, MMD
from dataset import get_bars_and_stripes
from plotting import *

import json


# Importing configuration file
with open("parameters.json", "r") as f:
    config = json.load(f)
SAMPLE_BITSTRING_DIMENSION = config['SAMPLE_BITSTRING_DIMENSION']
PRINT_TARGET_PDF = config['PRINT_TARGET_PDF']
DEVICE = config['DEVICE'] #cpu or gpu string format


jax.config.update("jax_platform_name", DEVICE)


# Setting circuit parameters
np.random.seed(42)
n_qubits = SAMPLE_BITSTRING_DIMENSION


# Output configuration parameters to user
print()
print("Configuration:")
print(f"bitsting samples dimension: {SAMPLE_BITSTRING_DIMENSION}")
if math.sqrt(SAMPLE_BITSTRING_DIMENSION).is_integer() == False:
    raise ValueError("bitstring samples dimension must be a perfect square!")
print(f"number of different samples:{2**(int(math.sqrt(SAMPLE_BITSTRING_DIMENSION)))*2-2}")
print(f"print target probability distribution: {PRINT_TARGET_PDF}")
print(f"Using: {jax.devices()}")
print()
  

def get_distribution(data):
    bitstrings = []
    nums = []
    for d in data:
        bitstrings += ["".join(str(int(i)) for i in d)]
        nums += [int(bitstrings[-1], 2)]
    py = np.zeros(2**SAMPLE_BITSTRING_DIMENSION)
    py[nums] = 1 / len(data)
    return py


def main():
    # Building dataset
    data = get_bars_and_stripes(int(math.sqrt(SAMPLE_BITSTRING_DIMENSION)))
    if PRINT_TARGET_PDF == True:
        print_bitstring_distribution(data)
    py = get_distribution(data)


    # Initializing MPS
    number_open_index = 8    # 8 blocchi di gate
    bond_dimension =   2**8  # rappresenta esattamente stati fino a 16 qubit
    psi = qtn.MPS_rand_state(L=number_open_index, bond_dim=bond_dimension)

    
    # Building Born Machine and MMD
    bandwidth = jnp.array([0.25, 0.5, 1])
    space = jnp.arange(2**n_qubits)
    mmd = MMD(bandwidth, space)


    # Optimization
    def sample_tn(psi):
        samples = []
        for b in psi.sample(512, seed=51,backend="jax"):
            samples.append(b[0])
        return samples

    def loss_fn(psi, py, mmd):
        samples = sample_tn(psi)
        px = get_distribution(samples)
        loss = mmd(px,py)
        return loss
    
    print(f"Initial loss value {loss_fn(psi,py,mmd)}")
    tnopt = qtn.TNOptimizer(
        psi,
        loss_fn = loss_fn,
        loss_kwargs={"py": py,"mmd":mmd},
        optimizer="adam",
        autodiff_backend= "AUTO"
    )
    psi_opt = tnopt.optimize(10)


if __name__ == "__main__":
    main()
