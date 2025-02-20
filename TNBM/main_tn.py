#
# DA FARE: 
# Ricorda! E' un modello implicito, quindi non conosciamo la distribuzione target (nell'esempio toy facciamo finta di non conoscerla)
# Bisogna aggiustare la MMD. Al momento è come quella di di Oriel con exact True, ci serve l'altra, cioè quella per i sample.
# Il training non si può fare con autograd così com'è. Bisogna fare un training loop e fare i gradienti con parameter shift.
# Bisogna capire come si fa a rendere la bond dimension una roba addestrabile
# 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
import math
import quimb as qu
import jax
import jax.numpy as jnp
import quimb as qu
import quimb.tensor as qtn

from  QCBM.born_machine import QCBM, MMD
from fun.dataset import get_bars_and_stripes
from fun.plotting import *

#
# Importing configuration file
# DEVICE = "cpu" or "gpu" (strings)
#
with open("parameters.json", "r") as f:
    config = json.load(f)
SAMPLE_BITSTRING_DIMENSION = config['SAMPLE_BITSTRING_DIMENSION']
PRINT_TARGET_PDF = config['PRINT_TARGET_PDF']
DEVICE = config['DEVICE']
EPOCHS = config['EPOCHS']
jax.config.update("jax_platform_name", DEVICE)
np.random.seed(42)
n_qubits = SAMPLE_BITSTRING_DIMENSION
print()
print("Configuration:")
print(f"bitsting samples dimension: {SAMPLE_BITSTRING_DIMENSION}")
if math.sqrt(SAMPLE_BITSTRING_DIMENSION).is_integer() == False:
    raise ValueError("bitstring samples dimension must be a perfect square!")
print(f"number of different samples:{2**(int(math.sqrt(SAMPLE_BITSTRING_DIMENSION)))*2-2}")
print(f"print target probability distribution: {PRINT_TARGET_PDF}")
print(f"Using: {jax.devices()}")
print()


def main():
    # 
    # Building dataset. We will then provide the model one data at the time and calculate MMD for each sample
    #
    data = get_bars_and_stripes(int(math.sqrt(SAMPLE_BITSTRING_DIMENSION)))
    if PRINT_TARGET_PDF == True:
        print_bitstring_distribution(data)

    #
    # MPS initialization. The number of open indexes controls the number of nodes in the MPS, and thus the number of 
    # gate blocks. Bond_dimension represent maximum bond dimension, that controls the expressivity of the model.
    # bond_dimension = 2^(D/2) where N is the number of qubits of the largest state exactly representable. 
    #
    number_open_index = 2
    bond_dimension =   2**8
    psi = qtn.MPS_rand_state(L=number_open_index, bond_dim=bond_dimension)

    #
    # Building Born Machine and MMD
    #
    bandwidth = jnp.array([0.25, 0.5, 1])
    space = jnp.arange(2**n_qubits)
    mmd = MMD(bandwidth, space)

    #
    # OPTIMIZATION
    # We have to understand which are our parameters. We have to give a fitting format to data to be given to tn.
    # At every loop of r we change learning rate for convergence.
    # At every epoch we wanto to compute gradients to update the MPS. We calculate the MMD loss inside the gradient computation.
    #
    rep = 10
    lr0 = 0.1
    
    np.random.seed(seed)
    parameters = np.random.normal(0, np.pi, size  = n_param)
    parameters[1:] = 0
    optimizer = ADAM(n_param, lr = lr0)
    seed = 12+42 + 53*(1+r)

    for ep in range(EPOCHS):

        lr = lr0*np.exp(-0.005*ep)
        lr = max(10**-6,lr)
        optimizer._lr = lr
        np.random.shuffle(target)
        target_train = target[:batch_size,...]

        median_grad = []
        global intermediary_tv
        intermediary_tv = []

        def loss(x):
            #
            # MMD loss function #
            #
            state_raw = qulacs.QuantumState(qubits)
            state_raw = ansatz(state_raw, x)
            samples = compute_samples(state_raw, exact = exact, n_shots=n_shots,
                                    n_qubits=qubits, projectors=projectors,values=values)
            loss =  MMD(samples,target_train,signal,exact)
            loss = np.mean(loss)

            return float(loss)

        for _ in range(grad_batch):

            gradients = compute_gradient(ansatz, parameters, target_train, qubits, n_shots,grad_loss_function, signal = signal, exact = exact, values = values)
            median_grad.append(gradients)

        temp_grad = np.mean(np.mean(np.array(median_grad), axis=0), axis=-1).reshape(-1)
        parameters = optimizer.update(parameters,temp_grad)

if __name__ == "__main__":
    main()