import json
import numpy as np
import matplotlib.pyplot as plt
import jax
import math
from sklearn.metrics import pairwise_distances
import quimb
import quimb.tensor as qtn

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

def A(n,l):
    # Define the set A of all possible bitstrings of lenght n and return the ones with norm l
    A = [np.array(list(bin(i)[2:].zfill(n)), dtype=int) for i in range(2**n)]
    A = [a for a in A if sum(a) == l]
    return A

# Define Dl string of observables
def Ommd(n, sigma):
    """
    Building the observable Dl for a given sigma and n. The standard name for the indexes are k1, k2, ..., kn, b1, b2, ..., bn.
    """
    # Set parameters:
    n = 9             # number of qubits per half
    L = 2 * n         # total number of sites

    Dl_list = []
    # Questa è la sommatoria di tutti i Dl con il loro coefficiente coef1
    for l in range(1, n+1):
        p_sigma = (1 - np.exp(-1/(2*sigma)))/2
        coef = p_sigma**l * (1-p_sigma)**(n-l)

        A_l = A(n, l)

        # Building D2l
        mpo_list = []
        for i in A_l:
            site1 = i
            site2 = site1 + n

            # Define operators:
            Z = np.array([[1, 0],
                        [0, -1]])
            I = np.eye(2)

            # Build MPO tensors: each tensor is shaped (1, 1, 2, 2)
            mpo_tensors = []
            for site in range(L):
                # Choose Z on the designated sites, I elsewhere:
                op = Z if site in site1 or site in site2 else I
                tensor = op.reshape(1, 2, 2) if site in [0, L - 1] else op.reshape(1, 1, 2, 2)
                mpo_tensors.append(tensor)

            # Create the MPO. Here, 'sites' and 'L' help label the tensor network.
            mpo = qtn.MatrixProductOperator(
                mpo_tensors,
                sites=range(L),
                L=L,
                shape='lrud'
            )
            mpo_list.append(mpo)

        # Sum all the MPOs
        Dl = mpo_list[0]
        for i in mpo_list[1:]:
            Dl = Dl.add_MPO(i)
        
        Dl_list.append(coef*Dl)

    O = Dl_list[0]
    for i in Dl_list[1:]:
        Dl = Dl.add_MPO(i)
    
    return Dl

def MMD(x, y, sigma, number_open_index, bond_dimension):
    """
    samples and target are two Matrix Product States.
    """

    x /= x.H @ x
    y /= y.H @ y
    rename_dict = {f'k{i}': f'k{i+number_open_index}' for i in range(number_open_index)}
    y.reindex_(rename_dict)

    """
    Building the MMD MPO. The default open indexes are k1, k2, ..., kn, b1, b2, ..., bn.
    Then we can contract the MPO with the MPS and the bitstring state to get the loss function.
    """

    Omm = Ommd(number_open_index, sigma)
    loss = x & Omm & y
    # Here we should do the trace but we have a MPS, what happends then??
    loss = loss @ loss.H

    return loss
