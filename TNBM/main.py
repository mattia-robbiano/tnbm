import sys
import pickle
import numpy as np
import quimb.tensor as qtn
sys.path.append('..')
from functions import *
from jax import config
jax.config.update("jax_enable_x64", True)

# TODO Parameters to be loaded from a json file
plot_opt = True
savetn_opt = True

def builder_mpo_loss(method, sigma, dimension):
    """ Define kernel MPO, all tensors are 2x2 matrices node_matrix, depending on the loss function user wants  to use
    """
    if method == 'lqf':
            node_matrix = np.array([[1, 0], [0, (dimension-1)/dimension]])
            node_matrix /= np.linalg.norm(node_matrix)
    elif method == 'mmd':
            node_matrix = np.array([[1, np.exp(-(1/(2*sigma**2)))], [np.exp(-(1/(2*sigma**2))), 1]])
            node_matrix /= np.linalg.norm(node_matrix)
    elif method == 'dkl':
        node_matrix = np.array([[1, 0], [0, 1]])
    
    tensors = [qtn.Tensor(data=node_matrix, inds=(f'cbase{i}', f'k{i}'), tags=f'kernel{i}') for i in range(dimension)]
    mpo = qtn.TensorNetwork(tensors)

    return mpo
    
def builder_mps_dataset(dimension, training_dataset, hyper = True):
    """Build the dataset as a list of Matrix Product States (MPS) or as a hyperindexed tensor network.

    Parameters:
    dimension (int): The dimension of the dataset.
    training_dataset (str): The type of training dataset to generate. Options are "cardinality" or "BS".
    hyper (bool, optional): If True, the dataset is represented as a hyperindexed tensor network. 
                            If False, the dataset is represented as a list of MPS. Default is True.
    
    Returns:
    tensor_network_dataset: The dataset represented either as a hyperindexed tensor network or a list of MPS.
    
    Raises:
    ValueError: If the training_dataset is not "cardinality" or "BS".
                If the dimension for "BS" dataset is not a perfect square.
    
    Notes:
    - For "cardinality" dataset, it generates a dataset based on cardinality.
    - For "BS" dataset, it generates a Bars and Stripes dataset.
    - When hyper is True, the dataset is converted into a hyperindexed tensor network.
    - When hyper is False, the dataset is converted into a list of MPS.
    """
    if training_dataset == "cardinality":
        dataset = get_cardinality(dimension, 200, int(dimension/2) - 1)

    elif training_dataset == "BS":
        dataset = get_bars_and_stripes(int(np.sqrt(dimension)))
        if math.sqrt(dimension).is_integer() == False:
            raise ValueError("bitstring samples dimension must be a perfect square!")
        
    else:
        raise ValueError("dataset not available in defaults: " + training_dataset)
    
    if hyper:
        """ In this case the dataset is a list of hyperindexed quimb tensor networks.
            create a hyperindexed representation of training set as a quimb tensor network
            keep in mind that [0,1] means measuring |0> and |1> on the two qubits. The elements of computational basis
            can be represented by vectors (0,1) and (1,0) respectively.
            The training set is represented by an MPO. Each node represent all the qubit measurement as a matrix.
            Each row of the matrix is single mesurements of the qubit in computational basis (0,1) or (1,0)
            We will have as many rows as the number of data points in the training set.
        """
        measurements = [[data[i] for data in dataset] for i in range(dimension)]
        tensor_data = [np.array([[1, 0] if m == 0 else [0, 1] for m in meas]) for meas in measurements]
        tensors = [qtn.Tensor(data=tensor_data[i], inds=('hyper', f'cbase{i}'), tags=f'sample{i}') for i in range(dimension)]
        tensor_network_dataset = qtn.TensorNetwork(tensors)
        for i in range(dimension):
            x = tensor_network_dataset.isel({'hyper': i}) 
            x = x/x.norm()
    
    else:
        """ In this case the dataset is a list of bitstrings, we will convert it to a list of MPS
        """
        tensor_network_dataset = []
        for data in dataset:
            state = qtn.MPS_computational_state(data)
            state /= state.norm()
            tensor_network_dataset.append(state)

    return tensor_network_dataset

def loss_fn(psi, training_tensor_network, kernel, method):
    """Computes the loss function to be minimized, which can be either Maximum Mean Discrepancy (MMD) or 
    Kullback-Leibler divergence (DKL).

    Parameters:
    psi (TensorNetwork): The tensor network representing the state psi.
    training_tensor_network (TensorNetwork): The tensor network representing the training data.
    kernel (TensorNetwork): The kernel tensor network used for computing the loss.
    method (str): The method to use for computing the loss. It can be either "mmd" for Maximum Mean Discrepancy 
                  or "dkl" for Kullback-Leibler divergence.
    
    Returns:
    float: The computed loss value.
    
    Raises:
    ValueError: If the provided method is not "mmd" or "dkl".
    
    Notes:
    - For the "mmd" method, the loss is computed as the Maximum Mean Discrepancy between the psi state and the 
      training data, where loss is implemented via a kernel tensor network, to be calculated externally.
    - For the "dkl" method, the loss is computed as the Kullback-Leibler divergence between the psi state and 
      the training data, where the loss is reconducted to the sum of the logs of the inner products between the psi
      state and the data states. Basically a Fidelity loss.
    """
    n = psi.L
    
    if method == "mmd":
        psi_copy = psi.copy()
        rename_dict = {f'k{i}': f'cbase{i}' for i in range(n)}
        psi_copy.reindex_(rename_dict)

        training_tensor_network_copy = training_tensor_network.copy()
        rename_dict = {f'k{i}': f'cbase{i}' for i in range(n)}
        training_tensor_network_copy.reindex_(rename_dict)

        mix_term = (psi & kernel & training_tensor_network).contract(output_inds = [], optimize = 'auto-hq')
        homogeneous_term_q = (psi & kernel & psi_copy).contract(output_inds = [], optimize = 'auto-hq')
        homogeneous_term_p = (training_tensor_network & kernel & training_tensor_network_copy).contract(output_inds = [], optimize = 'auto-hq')
        loss_value = homogeneous_term_q -2*mix_term  + homogeneous_term_p

    elif method == "dkl":
        loss_value = 0
        for i in range(training_tensor_network.ind_size('hyper')):
            x = training_tensor_network.isel({'hyper': i}).copy()
            qx = abs((psi & x).contract(output_inds = [], optimize = 'auto-hq'))**2+ 10**-8
            px = 1/training_tensor_network.ind_size('hyper')+ 10**-8
            loss_value += px*jnp.log(px/qx)
        loss_value = loss_value
            
    else:
        raise ValueError("loss function not available: "+method)

    return loss_value

def norm_fn(psi):
    psi /= psi.norm()
    return psi

def main():
    """Main function to execute the training or variance computation based on the mode specified in the parameters.
    The function performs the following tasks:
    1. Loads parameters from a JSON file.
    2. Depending on the mode ('training' or 'variance'), it either:
       - Trains a tensor network using the specified loss function and optimization method.
       - Computes the variance of the loss function for different bond dimensions and number of qubits.

    Returns:
    - If mode is 'training' and savetn_opt is False, returns the optimized tensor network (psi_opt).
    - Otherwise, saves the results to files and does not return any value.
    
    Raises:
    - Any exceptions raised by the underlying functions such as load_parameters, builder_mps_dataset, builder_mpo_loss, and loss_fn.
    
    Notes:
    - The function assumes the existence of certain files like 'parameters.json' and 'variance_results.txt'.
    - TODO The function, for now, also assumes the presence of certain global variables like plot_opt and savetn_opt, makes sense to put them in jason? .
"""

    sample_bitstring_dimension, mode_dataset, epochs, loss, mode, bond_dimension, sigma, fidelity_opt  = load_parameters("parameters.json", verbose = False)

    if mode == "training":

        training_tensor_network = builder_mps_dataset(dimension = sample_bitstring_dimension, training_dataset=mode_dataset, hyper=True)

        psi = qtn.MPS_rand_state(sample_bitstring_dimension, bond_dim=bond_dimension)
        for i, tensor in enumerate(psi):
            tensor.add_tag(f'psi{i}')
        psi /= psi.norm()

        kernel = builder_mpo_loss(method= loss, sigma= sigma, dimension= 
        sample_bitstring_dimension)
        
        # for i in range(training_tensor_network.ind_size('hyper')):
        #     x = training_tensor_network.isel({'hyper': i})
        #     if x.norm() != 1:
        #         print(f"Norm of x {i} is {x.norm()}")    
        # sys.exit()    

########### Optimization ###########
        def FidelityCallback(tnopt):
            state = tnopt.get_tn_opt()
            fidelity_test = loss_fn(state, training_tensor_network, kernel, method='dkl')
            with open('fidelity_test.txt', 'a') as f:
                f.write(f"Fidelity test: {fidelity_test}\n")

        tnopt = qtn.TNOptimizer(
            # the tensor network we want to optimize
            tn = psi,
            # the functions specfying the loss and normalization
            loss_fn=loss_fn,
            norm_fn=norm_fn,
            # we specify constants so that the arguments can be converted
            # to the  desired autodiff backend automatically
            loss_constants={"training_tensor_network": training_tensor_network, "kernel": kernel},
            # options
            loss_kwargs={"method": loss},
            # the underlying algorithm to use for the optimization
            # 'l-bfgs-b' is the default and often good for fast initial progress
            optimizer="adam",
            # which gradient computation backend to use
            autodiff_backend="jax",
            # callback function to execute after every optimization step
            callback=FidelityCallback if fidelity_opt == True else None,
        )

        psi_opt = tnopt.optimize(epochs)
        
        print(psi_opt.norm())
        print()

        if plot_opt == True:
            fig, ax = tnopt.plot()
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            fig.savefig("plot.png", facecolor='white')

        with open('tensor_network.pkl', 'wb') as f:
            pickle.dump(psi_opt, f)

    elif mode == "variance":
        with open('variance_results.txt', 'a') as f:
            bond_dimension_list = [2, 100, 600]
            for b in bond_dimension_list:
                for n in range(2, 20):
                    # Build the training set MPS with standard cardinality dataset
                    training_tensor_network = builder_mps_dataset(dimension=n, training_dataset=mode_dataset, hyper=True)
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
                    f.write(f'Variance for n = {n} and bond dimension {b} is {variance}\n')

if __name__ == "__main__":
    main()
