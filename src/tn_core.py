import numpy as np
import quimb.tensor as qtn
import jax.numpy as jnp

from synthetic_datasets import get_cardinality, get_bars_and_stripes

def builder_mpo_loss(method, sigma, dimension):
    """ Define kernel MPO, all tensors are 2x2 matrices node_matrix, depending on the loss function user wants  to use
    """
    if method == 'lqf': node_matrix = np.array([[1, 0], [0, (dimension-1)/dimension]])
    elif method == 'mmd': node_matrix = np.array([[1, np.exp(-(1/(2*sigma**2)))], [np.exp(-(1/(2*sigma**2))), 1]])
    else: raise ValueError("loss function not available: " + method)
    
    node_matrix = node_matrix.astype(float)
    node_matrix /= np.linalg.norm(node_matrix)
    
    tensors = [qtn.Tensor(data=node_matrix, inds=(f'cbase{i}', f'k{i}'), tags=f'kernel{i}') for i in range(dimension)]
    mpo = qtn.TensorNetwork(tensors)

    return mpo

def builder_mps_dataset(dimension, training_dataset):
    """Build the dataset as a list of Matrix Product States (MPS) or as a hyperindexed
    tensor network.The dataset is a list of hyperindexed quimb tensor networks.
    create a hyperindexed representation of training set as a quimb tensor network
    keep in mind that [0,1] means measuring |0> and |1> on the two qubits. The elements of computational basis
    can be represented by vectors (0,1) and (1,0) respectively.
    The training set is represented by an MPO. Each node represent all the qubit measurement as a matrix.
    Each row of the matrix is single mesurements of the qubit in computational basis (0,1) or (1,0)
    We will have as many rows as the number of data points in the training set.


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

    if training_dataset not in ["cardinality", "BS"]: raise ValueError(f"Dataset not available in defaults")
    if training_dataset == "cardinality": dataset = get_cardinality(dimension, 200, int(dimension / 2) - 1)
    else: dataset = get_bars_and_stripes(int(np.sqrt(dimension)))
    
    measurements = [[data[i] for data in dataset] for i in range(dataset.shape[0])]
    tensor_data = [jnp.array([[1, 0] if m == 0 else [0, 1] for m in meas]) for meas in measurements]
    tensors = [qtn.Tensor(data=tensor_data[i], inds=('hyper', f'cbase{i}'), tags=f'sample{i}') for i in range(dimension)]
    tensor_network_dataset = qtn.TensorNetwork(tensors)
    for i in range(dimension):
        x = tensor_network_dataset.isel({'hyper': i}) 
        x = x/x.norm() # normalize each tensor network corresponding to a data point
    
    return tensor_network_dataset