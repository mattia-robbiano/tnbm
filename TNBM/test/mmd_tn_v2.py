import sys
import quimb
import quimb.tensor as qtn
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('..')
from functions import *


"""create a hyperindexed representation of training set as a quimb tensor network
keep in mind that [0,1] means measuring |0> and |1> on the two qubits. The elements of computational basis
can be represented by vectors (0,1) and (1,0) respectively.
The training set is represented by an MPO. Each node represent all the qubit measurement as a matrix.
Each row of the matrix is single mesurements of the qubit in computational basis (0,1) or (1,0)
We will have as many rows as the number of data points in the training set.

In our example we will have 2 tensors, each with 4 rows and 2 columns.
"""
n = 9
dataset = get_bars_and_stripes(int(np.sqrt(n)))
measurements = [[data[i] for data in dataset] for i in range(n)]
training_tensor_data = [np.array([[1, 0] if m == 0 else [0, 1] for m in meas]) for meas in measurements]
training_tensors = [qtn.Tensor(data=training_tensor_data[i], inds=('hyper', f'cbase{i}'), tags=f'sample{i}') for i in range(n)]
training_tensor_network = qtn.TensorNetwork(training_tensors)
training_tensor_network/=training_tensor_network.norm()


""" Define kernel MPO, all tensors are 2x2 matrices node_matrix
"""
sigma = 0.09
node_matrix = np.array([[1, np.exp(-(1/(2*sigma**2)))], [np.exp(-(1/(2*sigma**2))), 1]])
#normailze
node_matrix /= np.linalg.norm(node_matrix)
kernel_tensors = [qtn.Tensor(data=node_matrix, inds=(f'cbase{i}', f'k{i}'), tags=f'kernel{i}') for i in range(n)]
kernel = qtn.TensorNetwork(kernel_tensors)


for _ in range(100):
    """ Initializing psi as MPS to be trained
    """
    psi = qtn.MPS_rand_state(n, bond_dim=8)
    for i, tensor in enumerate(psi):
        tensor.add_tag(f'psi{i}')

    """ Testing
    """
    mix_term = (psi & kernel & training_tensor_network).contract(output_inds = [], optimize = 'auto-hq')
    homogeneous_term_q = (psi & kernel & psi.H).contract(output_inds = [], optimize = 'auto-hq')
    homogeneous_term_p = (training_tensor_network & kernel & training_tensor_network.H).contract(output_inds = [], optimize = 'auto-hq')
    mmd = homogeneous_term_q -2*mix_term  + homogeneous_term_p
    #draw_mmd_tensor_network(..., n)
    print(mmd)