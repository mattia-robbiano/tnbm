"""
DA FARE: 

"""
import sys
import json
import numpy as np
import math
import quimb as qu
import jax
import jax.numpy as jnp
import quimb as qu
import quimb.tensor as qtn

from functions import *


PATH = "parameters.json"
SAMPLE_BITSTRING_DIMENSION, PRINT_TARGET_PDF, DEVICE, EPOCHS = load_parameters(PATH)
jax.config.update("jax_platform_name", DEVICE)
np.random.seed(42)
n_qubits = SAMPLE_BITSTRING_DIMENSION


"""
Building dataset. We will then provide the model one data at the time and calculate MMD for each sample
"""
dataset = get_bars_and_stripes(int(math.sqrt(SAMPLE_BITSTRING_DIMENSION)))
if PRINT_TARGET_PDF == True:
    print_bitstring_distribution(dataset)


"""
MPS initialization. The number of open indexes controls the number of nodes in the MPS, and thus the number of 
gate blocks. Bond_dimension represent maximum bond dimension, that controls the expressivity of the model.
bond_dimension = 2^(D/2) where N is the number of qubits of the largest state exactly representable.
TO DO: bond dimension should be some kind of variable
"""
number_open_index = 9
bond_dimension =   2
psi = qtn.MPS_rand_state(L=number_open_index, bond_dim=bond_dimension)
print("MODEL:")
print(psi)
print()
initial_tensor_array = []
for site, tensor in psi.tensor_map.items():
    initial_tensor_array.append(tensor.data)


"""
    OPTIMIZATION
    We have to understand which are our parameters. We have to give a fitting format to data to be given to tn.
    At every loop of r we change learning rate for convergence.
    At every epoch we wanto to compute gradients to update the MPS. We calculate the MMD loss inside the gradient computation.
"""

"""PARAMETERS INITIALIZATION"""
batch_size = 12
sigmas = np.array([0.25, 0.5, 1])
tolerance = 1e-6

"""LOADING TENSORS"""
tensor_array = []
for site, tensor in psi.tensor_map.items():
    tensor_array.append(tensor.data)


"""LOADING SAMPLE"""
np.random.shuffle(dataset)
target_train = dataset[:batch_size,...]
sample_generator = psi.sample(C=batch_size,seed=1234567)  
samples = np.array([bits for bits, _ in sample_generator])


"""COMPUTING LOSS FUNCTION FOR LOG"""
loss =  MMD(samples,target_train,sigmas)
loss = float(np.mean(loss))


"""COMPUTING GRADIENTS"""


