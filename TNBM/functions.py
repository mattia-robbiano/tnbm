import numpy as np
import matplotlib.pyplot as plt

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

class MMD:
#
# Define a MMD object, initialized once given the points on which we are evaluating exected value of kernel function
# and sigmas.
#
# METHODS:
#    
# __init__ ::
#    
#    scales: array of sigmas of the kernel functions we want to sum over
#    space:  array of points at which probability distributions will be evaluated when calculating expectation
#    value
#   
#    sq_dists: (space[:m None] - space[None, :]) the two 1d arrays are casted in a column and row matrices
#    subtracting, we are creating a matrix of pairwise distances (each squared by **2).
#
# k_expval ::
#    
#    returns expected value of K(x,y) over the distributions px, py, meaning we will extract x from px and
#    y from py
#
# __call__ :: returns the loss value, calculated with reduced expression, see google doc in readme for references    
#        
    def __init__(self, scales, space):
        gammas = 1 / (2 * (scales**2))
        sq_dists = np.abs(space[:, None] - space[None, :]) ** 2
        self.K = sum(np.exp(-gamma * sq_dists) for gamma in gammas) / len(scales)
        self.scales = scales

    def k_expval(self, px, py):
        return px @ self.K @ py

    def __call__(self, px, py):
        pxy = px - py
        return self.k_expval(pxy, pxy)
