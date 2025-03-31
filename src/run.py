import sys
import pickle
import numpy as np
from typing import Any

import jax
import jax.numpy as jnp
from jax import config
jax.config.update("jax_enable_x64", True)

import quimb.tensor as qtn

from metrics import MMD, KLD
from tn_core import builder_mps_dataset, builder_mpo_loss
import os

plot_opt = False
print_model = False

def write_to_file(filename: str, value: Any) -> None:
    """Write a value to a file, overwriting if the file already exists."""
    with open(filename, 'a') as file:
        file.write(f"{value}\n")

def norm_fn(psi):
    psi /= psi.norm()
    return psi

def run_training(bond_dimension, 
                 loss_fn, 
                 dataset_mode, epochs, callback_metrics: list[str], sigma=0.09):
    """
    Run the training process for the tensor network optimizer.
    Parameters:
    bond_dimension (int): The bond dimension for the MPS.
    loss_fn (function): The loss function to be used.
    dataset (tuple): A tuple containing the dataset name and the number of qubits.
    TODO: dataset must be a matrix of datapoints
    sigma (float): The standard deviation for the Gaussian kernel.
    TODO: sigma must be a list of standard deviations, and find a smarter way to pass it
    epochs (int): Number of epochs for training.
    callback_metrics (list[str]): List of metrics to be used in the callback.
    """

    for filename in ['loss.out', 'fidelity.out', 'mmd.out', 'kld.out']:
        if os.path.exists(filename):
            os.remove(filename)

    dataset, nqubit = dataset_mode

    training_tensor_network = builder_mps_dataset(dimension = nqubit, training_dataset=dataset)

    model = qtn.MPS_rand_state(nqubit, bond_dim=bond_dimension)
    for i, tensor in enumerate(model): tensor.add_tag(f'model{i}')
    model /= model.norm()

    node_kernel = jnp.array([[1, jnp.exp(-1/(2*sigma**2))], [jnp.exp(-1/(2*sigma**2)), 1]], dtype=float)
    #node_kernel /= jnp.linalg.norm(node_kernel)
    tensors_kernel = [qtn.Tensor(data=node_kernel, inds=(f'cbase{i}', f'k{i}'), tags=f'kernel{i}') for i in range(nqubit)]
    kernel = qtn.TensorNetwork(tensors_kernel)

    if print_model == True:
        rename_dict = {f'k{i}': f'cbase{i}' for i in range(model.L)}
        model.reindex_(rename_dict)
        print(model)
        print()
        print(training_tensor_network)
        print()
        fig = (model & training_tensor_network).draw(return_fig=True,)
        fig.patch.set_facecolor('white')
        fig.savefig('model_kernel_plot.pdf', facecolor='white')
        sys.exit()

    def callback_val(tnopt: Any) -> None:
        """Callback function to log optimization metrics."""
        
        state = tnopt.get_tn_opt().copy()
        state.reindex_({f'k{i}': f'cbase{i}' for i in range(nqubit)})

        write_to_file('loss.out', tnopt.loss)

        if "fidelity" in callback_metrics:
            normalization = training_tensor_network.ind_size('hyper')
            result = (state & training_tensor_network).contract(output_inds=[], optimize='auto-hq') / normalization
            write_to_file('fidelity.out', result)

        if "mmd" in callback_metrics:
            write_to_file('mmd.out', MMD(state, training_tensor_network, kernel))

        if "kld" in callback_metrics:
            write_to_file('kld.out', KLD(state, training_tensor_network, kernel))
    
    tnopt = qtn.TNOptimizer(
        tn = model,
        loss_fn=loss_fn,
        norm_fn=norm_fn,
        loss_constants={"training_tensor_network": training_tensor_network, "kernel": kernel},
        loss_kwargs={},
        optimizer="adam",
        autodiff_backend="jax",
        callback=callback_val,
        progbar=True
    )
    psi_opt = tnopt.optimize(epochs)
    
    fig, ax = tnopt.plot()
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    fig.savefig('plot.pdf', facecolor='white')
    with open('tensor_network.pkl', 'wb') as f: pickle.dump(psi_opt, f)

"""
TODO: to be fixed


def run_variance():
        with open('variance_results.txt', 'a') as f:
            bond_dimension_list = [2, 100, 600]
            for b in bond_dimension_list:
                for n in range(2, 20):
                    # Build the training set MPS with standard cardinality dataset
                    training_tensor_network = builder_mps_dataset(dimension=n, training_dataset=mode_dataset)
                    # Build the kernel MPO for the loss function
                    kernel = builder_mpo_loss(method=loss, sigma=sigma, dimension=n)

                    # Initialize the dataset state for n qubits, sample the psi state at random,
                    # and compute the variance of the loss function
                    loss_values = []
                    for k in range(100):
                        psi = qtn.MPS_rand_state(n, bond_dim=b, dist='uniform')
                        for i, tensor in enumerate(psi):
                            tensor.add_tag(f'psi{i}')
                        loss_values.append(loss_fn(psi, training_tensor_network, kernel, method=loss))

                    # Compute and log the variance of the loss function for the given bond dimension and number of qubits
                    variance = np.var(loss_values)
                    print(f"Variance for n = {n} and bond dimension {b} is {variance}")
                    f.write(f'Variance for n = {n} and bond dimension {b} is {variance}\n')"
"""