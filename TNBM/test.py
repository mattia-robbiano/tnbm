import sys
import pickle
import jax as jnp
import quimb.tensor as qtn
sys.path.append('..')
from functions import *
from main import loss_mpo_builder, dataset_mps_builder, loss_fn
jax.config.update("jax_enable_x64", True)

sample_bitstring_dimension, mode_dataset, device, epochs, loss, mode, bond_dimension, sigma = load_parameters("parameters.json", verbose = False)

n = sample_bitstring_dimension
training_tensor_network = dataset_mps_builder(dimension = n, default_dataset=mode_dataset, hyper=False, dataset=None)
psi = qtn.MPS_rand_state(n, bond_dim=bond_dimension)
for i, tensor in enumerate(psi):
    tensor.add_tag(f'psi{i}')

# print(training_tensor_network[0])
# print(psi)
kernel = loss_mpo_builder(loss= loss, sigma = sigma, dimension=n)

print(loss_fn(psi, training_tensor_network, kernel, method = "dkl"))