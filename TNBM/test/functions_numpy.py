import json
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import math
from sklearn.metrics import pairwise_distances
import quimb
import quimb.tensor as qtn

jax.config.update("jax_enable_x64", True)

def get_bars_and_stripes(n):

    bitstrings = [list(np.binary_repr(i, n))[::-1] for i in range(2**n)]
    bitstrings = np.array(bitstrings, dtype=int)
    
    stripes = bitstrings.copy()
    stripes = np.repeat(stripes, n, 0)
    stripes = stripes.reshape(2**n, n * n)

    bars = bitstrings.copy()
    bars = bars.reshape(2**n * n, 1)
    bars = np.repeat(bars, n, 1)
    bars = bars.reshape(2**n, n * n)

    dataset = np.vstack((stripes[0 : stripes.shape[0] - 1], bars[1 : bars.shape[0]]))

    return dataset


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
        probs = np.zeros(2**n)
        probs[nums] = 1 / len(data)
        
    plt.figure(figsize=(12, 5))
    plt.bar(np.arange(2**n), probs, width=2.0, label=r"$\pi(x)$")
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
    py = np.zeros(2**bitstring_dimension)
    py[nums] = 1 / len(data)

    return py

def load_parameters(path):
    #
    # Importing configuration file and IO
    # DEVICE = "cpu" or "gpu" (strings)
    #
    with open(path, "r") as f:
        config = json.load(f)
    SAMPLE_BITSTRING_DIMENSION = config['SAMPLE_BITSTRING_DIMENSION']
    PRINT_TARGET_PDF = config['PRINT_TARGET_PDF']
    DEVICE = config['DEVICE']
    EPOCHS = config['EPOCHS']

    jax.config.update("jax_platform_name", DEVICE)

    print()
    print("Configuration:")
    print(f"bitsting samples dimension: {SAMPLE_BITSTRING_DIMENSION}")
    if math.sqrt(SAMPLE_BITSTRING_DIMENSION).is_integer() == False:
        raise ValueError("bitstring samples dimension must be a perfect square!")
    print(f"number of different samples:{2**(int(math.sqrt(SAMPLE_BITSTRING_DIMENSION)))*2-2}")
    print(f"print target probability distribution: {PRINT_TARGET_PDF}")
    print(f"Using: {jax.devices()}")
    print()

    return SAMPLE_BITSTRING_DIMENSION, PRINT_TARGET_PDF, DEVICE, EPOCHS

def A(n,l):
    # Define the set A of all possible bitstrings of lenght n and return the ones with norm l
    A = [np.array(list(bin(i)[2:].zfill(n)), dtype=int) for i in range(2**n)]
    A = [a for a in A if sum(a) == l]
    return A

