#
# bozza, definire funzioni per ogni cosa e integrare con la versione quantum circuit
#
# In progress - building a Tensor Network Born Machine (TNBM).....ok
# 1 - defining random MPS with default quimb function.............ok
# 2 - try sampling from it
# 3 - define a loss function that yields a single real scalar
#
import quimb as qu
import quimb.tensor as qtn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import math

from  born_machine_1 import QCBM, MMD
from dataset import get_bars_and_stripes
from plotting import *

# Parameters
SAMPLE_BITSTRING_DIMENSION = 9
PRINT_TARGET_PDF = False
n_qubits = SAMPLE_BITSTRING_DIMENSION


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

    # Generating dataset
    data = get_bars_and_stripes(int(math.sqrt(SAMPLE_BITSTRING_DIMENSION)))
    if PRINT_TARGET_PDF == True:
        print_bitstring_distribution(data)  
    py = get_distribution(data)

    # Building initial MPS
    number_open_index = 8    # 8 blocchi di gate
    bond_dimension =   2**8  # rappresenta esattamente stati fino a 16 qubit
    psi = qtn.MPS_rand_state(L=number_open_index, bond_dim=bond_dimension)


    # Building MMD object, mmd is a loss function returning a single scalar value, thus, we can use automatic optimization method from Quimb
    bandwidth = jnp.array([0.25, 0.5, 1])
    space = jnp.arange(2**n_qubits)
    mmd = MMD(bandwidth, space)

    
    def loss_fn(psi, py):
        samples = []
        for b in psi.sample(512, seed=51):
            samples.append(b[0])
        px = get_distribution(samples)
        loss = mmd(px,py)
        return loss


    tnopt = qtn.TNOptimizer(
        psi,
        loss_fn=loss_fn,
        loss_constants={"py": py},
        optimizer="adam",
        autodiff_backend= "autograd"
    )

    #print(loss_fn(psi,py))
    psi_opt = tnopt.optimize(1000)

    
if __name__ == "__main__":
    main()
