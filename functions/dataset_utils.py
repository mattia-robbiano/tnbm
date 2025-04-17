import matplotlib.pyplot as plt
import numpy as np
import quimb.tensor as qtn
from tqdm import tqdm


def one_hot(x, dim = 2):
    """Given a list of labels, creates a one-hot representation of them"""
    m = len(x)
    y = np.zeros((m, dim), dtype=int)
    y[np.arange(m), x] = 1
    return y

######################################################################################
# DATASETS
######################################################################################

def bars_and_stripes(size: int, shuffle = True, max_samples = int(1e6)):
    """Creates the complete Bars and Stripes (BAS) dataset of given size."""
    
    # If possible, generate the full dataset (all possible bars and stripes)
    if 2 ** size < max_samples:
        data = np.array([np.array(list(np.binary_repr(i, width=size)), dtype=int) for i in range(2 ** size)])
    else:
        # Sample randomly some of them
        data = []
        for i in tqdm(list(range(max_samples))):
            rndi = np.random.randint(0, 2 ** size)
            data.append(list(np.binary_repr(rndi, width=size)))
        data = np.array(data, dtype=int)    
    
    # Tile vertically and horizontally
    bars = np.repeat(data[:, np.newaxis, :], size, axis=1)
    stripes = np.repeat(data[:, :, np.newaxis], size, axis=2)

    # If we generated the whole dataset, full white and full black 
    # images appear both in the bars and in the stripes. Removing them
    # from one of the two so that each image appears the same number of
    # times in the dataset.
    if 2 ** size < max_samples:
        bars = bars[1:len(bars)-1]
    
    # Combine them together
    bas = np.vstack([bars, stripes])

    if shuffle: 
        np.random.shuffle(bas)
    
    return bas

import numpy as np

def cardinality(state_dim, training_set_dim, cardinality, seed=0):
    """
    Generate a dataset of binary strings with a fixed number of ones (cardinality).

    Args:
        state_dim (int): Length of each binary string.
        training_set_dim (int): Number of samples (must be even).
        cardinality (int): Number of ones in each binary string.
        seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: Shuffled dataset of shape (training_set_dim, state_dim).
    """
    if training_set_dim % 2 != 0:
        raise ValueError("training_set_dim must be an even number")
    if not (0 <= cardinality <= state_dim):
        raise ValueError("cardinality must be between 0 and state_dim")

    rng = np.random.default_rng(seed)
    dataset = []

    for _ in range(training_set_dim):
        sample = np.zeros(state_dim, dtype=int)
        ones_indices = rng.choice(state_dim, size=cardinality, replace=False)
        sample[ones_indices] = 1
        dataset.append(sample)

    dataset = np.stack(dataset)
    rng.shuffle(dataset, axis=0)  # Shuffle dataset

    return dataset

def plot_binary_data(bas: np.ndarray):
    """Given a list of binary images, plots them all."""

    num_data = len(bas)
    rows = int(np.sqrt(num_data))
    fig, axs = plt.subplots(rows, rows, 
                            figsize=(10, 10),
                            sharex=True,
                            sharey=True)

    idx = 0
    for ax in axs:
        for ax_ in ax:
            ax_.imshow(bas[idx], 
                       cmap='gray',
                       vmin = 0, # for binary images
                       vmax = 1)
            idx += 1

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

######################################################################################
# TENSOR NETWORK REPRESENTATION OF THE DATASETS
######################################################################################
def hypertn_from_data(data):
    """Creates a hyper tensor network representation of the given dataset of size (num_samples, num_sites)."""

    num_samples = data.shape[0]
    rank = len(data.shape)
    if rank == 3:
        # bars and stripes like
        num_sites = data.shape[1] * data.shape[2]
    else:
        # cardinality like
        num_sites = data.shape[1]

    # reshape in the way every row is a feature vector
    data = data.reshape(num_samples, num_sites).T

    tensors = []
    for q, x in enumerate(data):
        # Transform bits to one-hot encoding
        t = qtn.Tensor(one_hot(x), 
                    inds = ['hyper', f'k{q}'], 
                    tags = [f'I{q}', 'DATA'])
        tensors.append(t)

    return qtn.TensorNetwork(tensors)

def mps_from_data(data):
    pass

