import json
import numpy as np
import matplotlib.pyplot as plt
import jax
import math
from sklearn.metrics import pairwise_distances

def kernel(X,Y,sigma):

    distances = pairwise_distances(X,Y,n_jobs=-1).reshape(-1)
    kern      = np.array([np.exp(-distances**2/(2*s)) for s in sigma])

    return np.mean(kern, axis=-1)

def MMD(samples, target, sigma):

    return kernel(samples,samples,sigma) - 2*kernel(samples,target,sigma) + kernel(target,target,sigma)

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

    return np.vstack((stripes[0 : stripes.shape[0] - 1], bars[1 : bars.shape[0]]))

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

class ADAM:
    def __init__(
        self,
        n_param,
        tol: float = 1e-6,
        lr: float = 1e-1,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        noise_factor: float = 1e-8,
        eps: float = 1e-10,
        amsgrad: bool = False,
    ) -> None:

        self._tol = tol
        self._lr = lr
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._noise_factor = noise_factor
        self._eps = eps
        self._amsgrad = amsgrad


        self._t = 1
        self._m = np.zeros((n_param))
        self._v = np.zeros((n_param))
        if self._amsgrad:
            self._v_eff = np.zeros((n_param))

    def update(self,params,derivative):

        self._m = self._beta_1 * self._m + (1 - self._beta_1) * derivative
        self._v = self._beta_2 * self._v + (1 - self._beta_2) * derivative * derivative
        lr_eff = self._lr * np.sqrt(1 - self._beta_2**self._t) / (1 - self._beta_1**self._t)
        if not self._amsgrad:
            params_new = params - lr_eff * self._m.flatten() / (
                    np.sqrt(self._v.flatten()) + self._noise_factor
                )
        else:
            self._v_eff = np.maximum(self._v_eff, self._v)
            params_new = params - lr_eff * self._m.flatten() / (
                    np.sqrt(self._v_eff.flatten()) + self._noise_factor
                )
        self._t +=1
        return params_new

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


    
