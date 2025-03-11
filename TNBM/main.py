import sys
import pickle
import quimb
import quimb.tensor as qtn
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('..')
from functions import *

def hom_mpo_builder(node_matrix, dimension):
            n = dimension
            node_matrix /= np.linalg.norm(node_matrix)
            kernel_tensors = [qtn.Tensor(data=node_matrix, inds=(f'cbase{i}', f'k{i}'), tags=f'kernel{i}') for i in range(n)]
            kernel = qtn.TensorNetwork(kernel_tensors)
            return kernel

def sample_mps_builder(dataset, dimension):
        n = dimension
        measurements = [[data[i] for data in dataset] for i in range(n)]
        training_tensor_data = [np.array([[1, 0] if m == 0 else [0, 1] for m in meas]) for meas in measurements]
        training_tensors = [qtn.Tensor(data=training_tensor_data[i], inds=('hyper', f'cbase{i}'), tags=f'sample{i}') for i in range(n)]
        training_tensor_network = qtn.TensorNetwork(training_tensors)
        training_tensor_network/=training_tensor_network.norm()
        return training_tensor_network

def loss_mpo_builder(loss, sigma, dimension):
    """ Define kernel MPO, all tensors are 2x2 matrices node_matrix, depending on the loss function user wants  to use
    """
    if loss == 'mmd' or loss == 'kqf':
        if loss == 'lqf':
            node_matrix = np.array([[1, 0], [0, (n-1)/n]])
        else:
            node_matrix = np.array([[1, np.exp(-(1/(2*sigma**2)))], [np.exp(-(1/(2*sigma**2))), 1]])
        
        return hom_mpo_builder(node_matrix, n)
    
    elif loss == 'dkl':
        print("DKL loss is not implemented yet")
        return
    
def mps_trainingset_builder(dimension, default_dataset):
    if default_dataset == "cardinality":
        dataset = get_cardinality(dimension, 200, int(dimension/2) - 1)
    elif default_dataset == "BS":
        dataset = get_bars_and_stripes(int(np.sqrt(dimension)))
        if math.sqrt(dimension).is_integer() == False:
            raise ValueError("bitstring samples dimension must be a perfect square!")
    else:
        raise ValueError("dataset not available: "+default_dataset)
    training_tensor_network = sample_mps_builder(dataset, dimension)

    return training_tensor_network

def loss_fn(psi,training_tensor_network,kernel, loss,n):
    """ Define loss function as model contraction
    """     
    psi_conj = psi.H
    rename_dict = {f'k{i}': f'cbase{i}' for i in range(n)}
    psi_conj.reindex_(rename_dict)

    training_tensor_network_conj = training_tensor_network.H
    rename_dict = {f'k{i}': f'cbase{i}' for i in range(n)}
    training_tensor_network_conj.reindex_(rename_dict)

    mix_term = (psi & kernel & training_tensor_network).contract(output_inds = [], optimize = 'auto-hq')
    homogeneous_term_q = (psi & kernel & psi_conj).contract(output_inds = [], optimize = 'auto-hq')
    homogeneous_term_p = (training_tensor_network & kernel & training_tensor_network_conj).contract(output_inds = [], optimize = 'auto-hq')
    if loss == 'mmd':
        mmd = homogeneous_term_q -2*mix_term  + homogeneous_term_p
        loss_value = mmd

    return loss_value


def main():
    """ Loading parameters"""
    sample_bitstring_dimension, mode_dataset, device, epochs, loss, mode, bond_dimension, sigma = load_parameters("parameters.json")


    """create a hyperindexed representation of training set as a quimb tensor network
    keep in mind that [0,1] means measuring |0> and |1> on the two qubits. The elements of computational basis
    can be represented by vectors (0,1) and (1,0) respectively.
    The training set is represented by an MPO. Each node represent all the qubit measurement as a matrix.
    Each row of the matrix is single mesurements of the qubit in computational basis (0,1) or (1,0)
    We will have as many rows as the number of data points in the training set.

    In our example we will have 2 tensors, each with 4 rows and 2 columns.
    """

    """ Building dataset
    """
    n = sample_bitstring_dimension
    training_tensor_network = mps_trainingset_builder(n, mode_dataset)


    """ Initializing psi as MPS to be trained
    """
    psi = qtn.MPS_rand_state(n, bond_dim=bond_dimension)
    for i, tensor in enumerate(psi):
        tensor.add_tag(f'psi{i}')


    if mode == "training":
        kernel = loss_mpo_builder(loss, sigma, n)

        tnopt = qtn.TNOptimizer(
            # the tensor network we want to optimize
            psi,

            # the functions specfying the loss and normalization
            loss_fn=loss_fn,

            # we specify constants so that the arguments can be converted
            # to the  desired autodiff backend automatically
            loss_constants={"training_tensor_network": training_tensor_network, "kernel": kernel},

            # options
            loss_kwargs={'loss': loss, 'n': n},

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
        with open('tensor_network.pkl', 'wb') as f:
            pickle.dump(psi_opt, f)

    elif mode == "variance":
        with open('variance_results.txt', 'a') as f:
            bond_dimension_list = [10, 100, 400, 600, 1000]

            for b in bond_dimension_list:
                for n in range(2, 20):


                    """ building the training set mps with standard cardinality dataset
                        and the kernel MPO for the loss function
                    """
                    training_tensor_network = mps_trainingset_builder(dimension=n, default_dataset = "cardinality")
                    kernel = loss_mpo_builder(loss="mmd", sigma=sigma, dimension = n)


                    """ once initialized the dataset state for n qubits, we can now sample the psi state
                        at random and compute the variance of the loss function
                    """
                    loss_values = []
                    for k in range(100):
                        psi = qtn.MPS_rand_state(n, bond_dim=b)
                        for i, tensor in enumerate(psi):
                            tensor.add_tag(f'psi{i}')
                        loss_values.append(loss_fn(psi,training_tensor_network,kernel, loss,n))
                    
                    """ computing variance of the loss function for the given bond dimension and number of qubits
                    """
                    variance = np.var(loss_values)
                    print(f"Variance for n = {n} and bond dimension {b} is {variance}")
                    f.write(f'Variance for n = {n} and bond dimension {b} is {variance}\n')

                
if __name__ == "__main__":
    main()
