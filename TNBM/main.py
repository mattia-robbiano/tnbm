import sys
import pickle
import numpy as np
import quimb.tensor as qtn
sys.path.append('..')
from functions import *
from jax import config
jax.config.update("jax_enable_x64", True)

# Parameters to be loaded from a json file soon :)
plot_opt = True
savetn_opt = True


def loss_mpo_builder(loss, sigma, dimension):
    """ Define kernel MPO, all tensors are 2x2 matrices node_matrix, depending on the loss function user wants  to use
    """
    if loss == 'lqf':
            node_matrix = np.array([[1, 0], [0, (dimension-1)/dimension]])
            node_matrix /= np.linalg.norm(node_matrix)
    elif loss == 'mmd':
            node_matrix = np.array([[1, np.exp(-(1/(2*sigma**2)))], [np.exp(-(1/(2*sigma**2))), 1]])
            node_matrix /= np.linalg.norm(node_matrix)
    elif loss == 'dkl':
        node_matrix = np.array([[1, 0], [0, 1]])
    
    tensors = [qtn.Tensor(data=node_matrix, inds=(f'cbase{i}', f'k{i}'), tags=f'kernel{i}') for i in range(dimension)]
    mpo = qtn.TensorNetwork(tensors)

    return mpo
    
def dataset_mps_builder(dimension, default_dataset="None", hyper = True, dataset=None):
    if default_dataset == "cardinality":
        dataset = get_cardinality(dimension, 200, int(dimension/2) - 1)
    elif default_dataset == "BS":
        dataset = get_bars_and_stripes(int(np.sqrt(dimension)))
        if math.sqrt(dimension).is_integer() == False:
            raise ValueError("bitstring samples dimension must be a perfect square!")
    elif default_dataset == "ising":
        dataset = get_ising(dimension,20)
    elif default_dataset == "None":
        if dataset is None:
            raise ValueError("dataset must be provided!")
    else:
        raise ValueError("dataset not available in defaults: "+default_dataset)
    
    if hyper:
        """ In this case the dataset is a list of hyperindexed quimb tensor networks
        """
        measurements = [[data[i] for data in dataset] for i in range(dimension)]
        tensor_data = [np.array([[1, 0] if m == 0 else [0, 1] for m in meas]) for meas in measurements]
        tensors = [qtn.Tensor(data=tensor_data[i], inds=('hyper', f'cbase{i}'), tags=f'sample{i}') for i in range(dimension)]
        tensor_network = qtn.TensorNetwork(tensors)
        tensor_network_dataset = tensor_network/tensor_network.norm()

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
    """ Define loss function as model contraction. Not adapted for DKL and LQF loss functions yet
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
        mmd = homogeneous_term_q -2*mix_term  + homogeneous_term_p
        loss_value = mmd

    elif method == "dkl":
        """ kernel here is an mpo with all tensors being identity matrices
        """
        psi /= psi.norm()
        q = 0
        for data in training_tensor_network:
            contr = (psi & data).contract(output_inds = [], optimize = 'auto-hq')
            q -= jnp.log(abs(contr)**2)

        loss_value = q / len(training_tensor_network)

    elif method == "lqf":
        #not implemented yet
        pass

    else:
        raise ValueError("loss function not available: "+method)

    return loss_value

def main():
    """ Loading parameters"""
    sample_bitstring_dimension, mode_dataset, device, epochs, loss, mode, bond_dimension, sigma = load_parameters("parameters.json", verbose = False)

    if mode == "training":
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
        training_tensor_network = dataset_mps_builder(dimension = n, default_dataset=mode_dataset, hyper=True, dataset=None)

        if loss == "dkl":
            for data in training_tensor_network:
                data.reindex_({f'cbase{i}': f'k{i}' for i in range(n)})
                data.add_tag('data')

        """ Initializing psi as MPS to be trained
        """
        psi = qtn.MPS_rand_state(n, bond_dim=bond_dimension)
        for i, tensor in enumerate(psi):
            tensor.add_tag(f'psi{i}')

        """ Building kernel MPO for the loss function
        """
        kernel = loss_mpo_builder(loss= loss, sigma= sigma, dimension= n)

        tnopt = qtn.TNOptimizer(
            # the tensor network we want to optimize
            psi,

            # the functions specfying the loss and normalization
            loss_fn=loss_fn,

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
        )

        psi_opt = tnopt.optimize(epochs)

        if plot_opt == True:
            fig, ax = tnopt.plot()
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            fig.savefig("plot.png", facecolor='white')

        if savetn_opt == True:
            with open('tensor_network.pkl', 'wb') as f:
                pickle.dump(psi_opt, f)
        else:
            return psi_opt


    elif mode == "variance":
        with open('variance_results.txt', 'a') as f:
            
            bond_dimension_list = [2, 100, 600]
            for b in bond_dimension_list:
                for n in range(2, 20):

                    """ building the training set mps with standard cardinality dataset
                        and the kernel MPO for the loss function
                    """
                    training_tensor_network = dataset_mps_builder(dimension=n, default_dataset="cardinality")
                    kernel = loss_mpo_builder(loss="mmd", sigma=sigma, dimension=n)

                    """ once initialized the dataset state for n qubits, we can now sample the psi state
                        at random and compute the variance of the loss function
                    """
                    loss_values = []
                    for k in range(100):
                        psi = qtn.MPS_rand_state(n, bond_dim=b, dist='uniform')
                        for i, tensor in enumerate(psi):
                            tensor.add_tag(f'psi{i}')
                        loss_values.append(loss_fn(psi, training_tensor_network, kernel))

                    """ computing variance of the loss function for the given bond dimension and number of qubits
                    """
                    variance = np.var(loss_values)
                    print(f"Variance for n = {n} and bond dimension {b} is {variance}")
                    f.write(f'Variance for n = {n} and bond dimension {b} is {variance}\n')

if __name__ == "__main__":
    main()
