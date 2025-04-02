import numpy as np
import jax
import jax.numpy as jnp
from jax import random

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
