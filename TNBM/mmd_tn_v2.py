import sys
import pickle
import quimb
import quimb.tensor as qtn
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('..')
from functions import *

def main():
    """create a hyperindexed representation of training set as a quimb tensor network
    keep in mind that [0,1] means measuring |0> and |1> on the two qubits. The elements of computational basis
    can be represented by vectors (0,1) and (1,0) respectively.
    The training set is represented by an MPO. Each node represent all the qubit measurement as a matrix.
    Each row of the matrix is single mesurements of the qubit in computational basis (0,1) or (1,0)
    We will have as many rows as the number of data points in the training set.

    In our example we will have 2 tensors, each with 4 rows and 2 columns.
    """
    n = 9
    dataset = get_GHZ(n, 100)

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


    """ Initializing psi as MPS to be trained
    """
    psi = qtn.MPS_rand_state(n, bond_dim=8)
    for i, tensor in enumerate(psi):
        tensor.add_tag(f'psi{i}')

    """ Testing
    """

    #draw_mmd_tensor_network(..., n)

    def loss_fn(psi,training_tensor_network,kernel):
        psi_conj = psi.H
        rename_dict = {f'k{i}': f'cbase{i}' for i in range(n)}
        psi_conj.reindex_(rename_dict)

        training_tensor_network_conj = training_tensor_network.H
        rename_dict = {f'k{i}': f'cbase{i}' for i in range(n)}
        training_tensor_network_conj.reindex_(rename_dict)

        mix_term = (psi & kernel & training_tensor_network).contract(output_inds = [], optimize = 'auto-hq')
        homogeneous_term_q = (psi & kernel & psi_conj).contract(output_inds = [], optimize = 'auto-hq')
        homogeneous_term_p = (training_tensor_network & kernel & training_tensor_network_conj).contract(output_inds = [], optimize = 'auto-hq')
        mmd = homogeneous_term_q -2*mix_term  + homogeneous_term_p

        return mmd
    

    tnopt = qtn.TNOptimizer(
        # the tensor network we want to optimize
        psi,

        # the functions specfying the loss and normalization
        loss_fn=loss_fn,

        # we specify constants so that the arguments can be converted
        # to the  desired autodiff backend automatically
        loss_constants={"training_tensor_network": training_tensor_network, "kernel": kernel},

        # the underlying algorithm to use for the optimization
        # 'l-bfgs-b' is the default and often good for fast initial progress
        optimizer="adam",

        # which gradient computation backend to use
        autodiff_backend="jax",
    )


    psi_opt = tnopt.optimize(3000)
    fig, ax = tnopt.plot()
    fig.patch.set_facecolor('white')  # Set figure background to white
    ax.set_facecolor('white')  # Set axes background to white
    fig.savefig("plot.png", facecolor='white')

    """
    Save the tensor network
    """
    with open('./tensor_network.pkl', 'wb') as f:
        pickle.dump(psi_opt, f)


if __name__ == "__main__":
    main()
