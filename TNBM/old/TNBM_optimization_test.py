import quimb
import quimb.tensor as qtn
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from functions import *

L = 3
D = 8
sigma = 0.09

# create a random MPS as our initial target to optimize
#psi = qtn.MPS_rand_state(L, bond_dim=D)
mpo = Ommd(L, sigma)
#mpo = qtn.MPO_identity(L)

dense_op = mpo.to_dense()  # Convert MPO to dense matrix
eigvals, eigvecs = np.linalg.eig(dense_op)
sorted_eigvals = np.sort(eigvals)
print("Eigenvalues in ascending order:", sorted_eigvals)