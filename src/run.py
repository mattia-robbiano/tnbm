import sys
import pickle
import numpy as np
from typing import Any
from tqdm import tqdm

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

def run_training(bond_dimension, loss_fn, dataset_mode, epochs, callback_metrics: list[str], sigma=0.09):
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
    node_kernel /= jnp.linalg.norm(node_kernel)
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

def compute_variance(bond_dim, dimension, loss_fn, training_tensor_network, kernel, iterations=100):
    """
    Compute the variance of loss values for a given bond dimension and dimension.

    This function generates a random Matrix Product State (MPS) with a specified 
    bond dimension and dimension, computes the loss values using the provided 
    loss function, and calculates the variance of these loss values over a 
    specified number of iterations.

    Args:
        bond_dim (int): The bond dimension of the MPS.
        dimension (int): The physical dimension of the MPS.
        loss_fn (Callable): A function that computes the loss given an MPS, 
            a training tensor network, and a kernel.
        training_tensor_network (Any): The training tensor network used in the 
            loss computation.
        kernel (Any): The kernel used in the loss computation.
        iterations (int, optional): The number of iterations to compute the 
            loss values. Defaults to 100.

    Returns:
        jnp.ndarray: The variance of the computed loss values.
    """
    loss_list = []
    for _ in range(iterations):
        psi = qtn.MPS_rand_state(dimension, bond_dim=bond_dim, dist='uniform')
        loss_list.append(loss_fn(psi, training_tensor_network, kernel))
    return np.var(loss_list)

def run_variance(bond_dimension: list[int], loss_fn, dataset_mode, sigma=0.09):
    """
    Computes and logs the variance for different bond dimensions and qubit counts.
    Args:
        bond_dimension (list[int]): A list of bond dimensions to iterate over.
        loss_fn (callable): The loss function to be used for variance computation.
        dataset_mode (tuple): A tuple containing the dataset and its mode.
        sigma (float, optional): The sigma value for the kernel function. Defaults to 0.09.
    Workflow:
        - Iterates over the provided bond dimensions.
        - For each bond dimension, iterates over qubit counts from 2 to 19.
        - Builds a training tensor network using the provided dataset and qubit count.
        - Constructs a kernel using the specified method, sigma, and qubit count.
        - Computes the variance using the bond dimension, qubit count, loss function, 
          training tensor network, and kernel.
        - Logs the variance to the console and writes it to an output file.
    Output:
        - Prints the variance for each bond dimension and qubit count.
        - Writes the variance to a file named `variance_bond_<bond_dimension>.out`.
    Note:
        Ensure that the functions `builder_mps_dataset`, `builder_mpo_loss`, 
        `compute_variance`, and `write_to_file` are defined and available in the 
        scope where this function is used.
    """
    dataset, _ = dataset_mode
    qubit_variance = []
    for nqubit in [2**2, 3**2, 4**2, 5**2]:
        bond_variance = []
        training_tensor_network = builder_mps_dataset(dimension=nqubit, training_dataset=dataset)
        kernel = builder_mpo_loss(method='mmd', sigma=sigma, dimension=nqubit)
        for b in bond_dimension:
            variance = compute_variance(b, nqubit, loss_fn, training_tensor_network, kernel)
            bond_variance.append(variance)
            print(f"Variance for bond dimension {b} and nqubit {nqubit}: {variance}")
        qubit_variance.append(bond_variance)

    np.save('variance.npy', qubit_variance)