import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.random as random
import math
from sklearn.metrics import pairwise_distances
import quimb
import quimb.tensor as qtn

jax.config.update("jax_enable_x64", True)

def get_cardinality(state_dim, training_set_dim, cardinality, seed=0):
    """
    Generate a dataset of binary strings with a fixed number of ones (cardinality).
    
    Args:
        state_dim (int): Length of each binary string.
        training_set_dim (int): Number of samples (must be even).
        cardinality (int): Number of ones in each binary string.
        seed (int): Random seed for reproducibility.

    Returns:
        jax.numpy.ndarray: Shuffled dataset of shape (training_set_dim, state_dim).
    """
    if training_set_dim % 2 != 0:
        raise ValueError("training_set_dim must be an even number")
    if not (0 <= cardinality <= state_dim):
        raise ValueError("cardinality must be between 0 and state_dim")

    key = random.PRNGKey(seed)
    dataset = []

    for i in range(training_set_dim):
        key, subkey = random.split(key)
        sample = jnp.zeros(state_dim, dtype=int)
        ones_indices = random.choice(subkey, state_dim, shape=(cardinality,), replace=False)
        sample = sample.at[ones_indices].set(1)
        dataset.append(sample)

    dataset = jnp.stack(dataset)
    dataset = random.permutation(key, dataset, axis=0)  # Shuffle dataset

    return dataset

def get_bars_and_stripes(n):

    bitstrings = [list(np.binary_repr(i, n))[::-1] for i in range(2**n)]
    bitstrings = jnp.array(bitstrings, dtype=int)
    
    stripes = bitstrings.copy()
    stripes = jnp.repeat(stripes, n, 0)
    stripes = stripes.reshape(2**n, n * n)

    bars = bitstrings.copy()
    bars = bars.reshape(2**n * n, 1)
    bars = jnp.repeat(bars, n, 1)
    bars = bars.reshape(2**n, n * n)

    dataset = jnp.vstack((stripes[0 : stripes.shape[0] - 1], bars[1 : bars.shape[0]]))

    return dataset

def get_GHZ(state_dim, training_set_dim):
    """
    Function to generate GHZ state of dimension state_dim and training set of dimension training_set_dim.
    The dataset will have half of the samples as all 0s and the other half as all 1s.
    """
    if training_set_dim % 2 != 0:
        raise ValueError("training_set_dim must be an even number")

    half_dim = training_set_dim // 2
    zeros = jnp.zeros((half_dim, state_dim))
    ones = jnp.ones((half_dim, state_dim))
    training_set = jnp.vstack((zeros, ones))
    training_set = jax.random.permutation(jax.random.PRNGKey(0), training_set, axis=0)

    return training_set

def print_bitstring_distribution(data):
#
# Simple function to lot histogram of occurence of bitstring in the dataset over all possible bitstring of that dimension
#
    n = data.shape[1]
    print(data.shape)
    bitstrings = []
    nums = []
    for d in data:
        bitstrings += ["".join(str(int(i)) for i in d)]
        nums += [int(bitstrings[-1], 2)]
        probs = jnp.zeros(2**n)
        probs[nums] = 1 / len(data)
        
    plt.figure(figsize=(12, 5))
    plt.bar(jnp.arange(2**n), probs, width=2.0, label=r"$\pi(x)$")
    plt.xticks(nums, bitstrings, rotation=80)
    plt.xlabel("Samples")
    plt.ylabel("Prob. Distribution")
    plt.legend(loc="upper right")
    plt.subplots_adjust(bottom=0.3)
    plt.show()

def get_distribution(data,bitstring_dimension):
    #
    # Simple function to plot discrete probability distribution of a bitstring dataset over all the possible bitstrings
    # with the same dimension.
    #
    bitstrings = []
    nums = []
    for d in data:
        bitstrings += ["".join(str(int(i)) for i in d)]
        nums += [int(bitstrings[-1], 2)]
    py = jnp.zeros(2**bitstring_dimension)
    py[nums] = 1 / len(data)

    return py

def load_parameters(path, verbose = True):
    """
    Importing configuration file and IO
    DEVICE = "cpu" or "gpu" (strings)
    """
    with open(path, "r") as f:
        config = json.load(f)
    SAMPLE_BITSTRING_DIMENSION = config['SAMPLE_BITSTRING_DIMENSION']
    MODE_DATASET = config['MODE_DATASET']
    DEVICE = config['DEVICE']
    EPOCHS = config['EPOCHS']
    LOSS = config['LOSS']
    MODE = config['MODE']
    BOND_DIMENSION = config['BOND_DIMENSION']
    SIGMA = config['SIGMA']

    jax.config.update("jax_platform_name", DEVICE)

    if verbose:
        print()
        print("Configuration:")
        print(f"bitsting samples dimension: {SAMPLE_BITSTRING_DIMENSION}")
        print(f"number of different samples:{2**(int(math.sqrt(SAMPLE_BITSTRING_DIMENSION)))*2-2}")
        print(f"Dataset: {MODE_DATASET}")
        print(f"Using: {jax.devices()}")
        print(f"Number of epochs: {EPOCHS}")
        print(f"Loss function: {LOSS}")
        print(f"Mode: {MODE}")
        print(f"Bond dimension: {BOND_DIMENSION}")
        print(f"Sigma: {SIGMA}")
        print()

    return SAMPLE_BITSTRING_DIMENSION, MODE_DATASET, DEVICE, EPOCHS, LOSS, MODE, BOND_DIMENSION, SIGMA

def A(n,l):
    """
    Simple function to define the set A of all possible bitstrings of lenght n and return the ones with norm l, where l is the number of 1s in the bitstring.
    """
    A = [np.array(list(bin(i)[2:].zfill(n)), dtype=int) for i in range(2**n)]
    A = [a for a in A if sum(a) == l]
    return A

def Ommd(n, sigma):
    """
    Building the observable Ommd. The observable is a sum of all the Dl operators, each one weighted by a coefficient.
    The Dl operator is a sum of all the MPOs that are the sum of all the Z operators on the sites of the bitstring.
    """

    #     L = 2 * n
    #     p_sigma = (1 - jnp.exp(-1/(2*sigma)))/2
    #     Z = jnp.array([[1, 0 ],
    #                    [0 ,-1]])
    #     Id = jnp.eye(2)

    #     """ 
    #     Sum over l from 1 to n, i am excluding 0 because i'm not sure how to treat it. For each l I am calculating the coefficient and the set A of all the bitstrings of length n with l 1s, (|A|=l)
    #     """
    #     Dl_list = []
    #     for l in range(1, n+1):
            
    #         coef = p_sigma**l * (1-p_sigma)**(n-l)
    #         A_l = A(n, l)

    #         """ 
    #         Loop over all bitstrings in the set A_l. i is the bitstring, site1 and site2 are the sets of indexes where the string
    #         of operators contains a Z acording two:
    #         D2l = sum_{i in A_l} tensor_product_{i in Al} Z(i)^Z(i+n), where ^ is kronecker product.
            
    #         PROBABLY PROBLEM HERE!
    #         """
    #         pauli_string_list = []
    #         for bitstring_array in A_l:
    #             for i in range(n):
    #                 if bitstring_array[i] == 1:
    #                     pauli_string_list.append(Id)
    #                 if bitstring_array[i] == 0:
    #                     pauli_string_list.append(Z)

    #             mpo_tensors = []
    #             for site in range(L):
    #                 op = Z if site in site1 or site in site2 else I
    #                 tensor = op.reshape(1, 2, 2) if site in [0, L - 1] else op.reshape(1, 1, 2, 2)
    #                 mpo_tensors.append(tensor)

    #             mpo = qtn.MatrixProductOperator(
    #                 mpo_tensors,
    #                 sites=range(L),
    #                 L=L,
    #                 shape='lrud'
    #             )
    #             mpo_list.append(mpo)

    #         # Sum all the MPOs
    #         Dl = mpo_list[0]
    #         for i in mpo_list[1:]:
    #             Dl = Dl.add_MPO(i)
            
    #         Dl_list.append(coef*Dl)

    #     O = Dl_list[0]
    #     for i in Dl_list[1:]:
    #         Dl = Dl.add_MPO(i)
        
    #     return Dl

def MMD(x, y,Ommd, sigma, number_open_index, bond_dimension):
    """
    x and y are two Matrix Product States.
    we want to compute the Tr(Ommd*(x^y)), since x and y are pure states, the expression semplfies as
    the expectation value of Ommd on x and y. We have to reindex the tensors to match the indexes of the MPO (k and b by
    default).

    Schematically the structure of the TN is:
    PSI and DATA            P-P-P-P-P-P D-D-D-D-D-D
                            | | | | | | | | | | | |        
    MPO                     O-O-O-O-O-O-O-O-O-O-O-O
                            | | | | | | | | | | | |
    PSI_CONJ and DATA_CONJ  p-p-p-p-p-p d-d-d-d-d-d

    """
    x /= x.H @ x
    y /= y.H @ y
    rename_dict = {f'k{i}': f'k{i+number_open_index}' for i in range(number_open_index)}
    y.reindex_(rename_dict)

    x_conj = x.H
    rename_dict = {f'k{i}': f'b{i}' for i in range(number_open_index)}
    x_conj.reindex_(rename_dict)

    y_conj = y.H
    rename_dict = {f'k{i}': f'b{i}' for i in range(number_open_index, 2*number_open_index)}
    y_conj.reindex_(rename_dict)

    loss = x & y & Ommd & x_conj.H & y_conj.H

    loss = loss^...

    return loss
