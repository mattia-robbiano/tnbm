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
    
    # Combine them together
    bas = np.vstack([bars, stripes])

    if shuffle: 
        np.random.shuffle(bas)
    
    return bas

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
            ax_.imshow(bas[idx], cmap='gray')
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
    num_sites = data.shape[1] * data.shape[2]

    # Flatten the images, and reshape to size (num_sites, num_samples)
    data = data.reshape(num_samples, num_sites).T

    tensors = []
    for q, x in enumerate(data):
        # Transform bits to one-hot encoding
        t = qtn.Tensor(one_hot(x), 
                    inds = ['hyper', f'k{q}'], 
                    tags = [f'I{q}', 'DATA'])
        tensors.append(t)

    return qtn.TensorNetwork(tensors)