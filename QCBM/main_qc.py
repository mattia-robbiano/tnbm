import json
import numpy as np
import math
import pennylane as qml
import jax
import jax.numpy as jnp
import optax
from  born_machine import QCBM, MMD
from fun.dataset import get_bars_and_stripes
from fun.plotting import *


# Importing configuration file
with open("parameters.json", "r") as f:
    config = json.load(f)
SAMPLE_BITSTRING_DIMENSION = config['SAMPLE_BITSTRING_DIMENSION']
PRINT_TARGET_PDF = config['PRINT_TARGET_PDF']
DEVICE = config['DEVICE'] #cpu or gpu string format


jax.config.update("jax_platform_name", DEVICE)


# Setting circuit parameters
np.random.seed(42)
n_qubits = SAMPLE_BITSTRING_DIMENSION
n_layers = 6 # to be moved in json
dev = qml.device("default.qubit", wires=n_qubits)


# Output configuration parameters to user
print()
print("Configuration:")
print(f"bitsting samples dimension: {SAMPLE_BITSTRING_DIMENSION}")
if math.sqrt(SAMPLE_BITSTRING_DIMENSION).is_integer() == False:
    raise ValueError("bitstring samples dimension must be a perfect square!")
print(f"number of different samples:{2**(int(math.sqrt(SAMPLE_BITSTRING_DIMENSION)))*2-2}")
print(f"print target probability distribution: {PRINT_TARGET_PDF}")
print(f"Using: {jax.devices()}")
print()

@qml.qnode(dev)
def circuit(weights):
    qml.StronglyEntanglingLayers(weights=weights, ranges=[1] * n_layers, wires=range(n_qubits))
    return qml.probs()

def get_distribution(data):
    bitstrings = []
    nums = []
    for d in data:
        bitstrings += ["".join(str(int(i)) for i in d)]
        nums += [int(bitstrings[-1], 2)]
    py = np.zeros(2**SAMPLE_BITSTRING_DIMENSION)
    py[nums] = 1 / len(data)
    return py


def main():
    # Building dataset
    data = get_bars_and_stripes(int(math.sqrt(SAMPLE_BITSTRING_DIMENSION)))
    if PRINT_TARGET_PDF == True:
        print_bitstring_distribution(data)
    py = get_distribution(data)


    # Initializing ansatz
    wshape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    weights = np.random.random(size=wshape)    
    jit_circuit = jax.jit(circuit)

    
    # Building Born Machine and MMD
    bandwidth = jnp.array([0.25, 0.5, 1])
    space = jnp.arange(2**n_qubits)
    mmd = MMD(bandwidth, space)
    qcbm = QCBM(jit_circuit, mmd, py)

    
    # Setting up optimization. In update step we also print kl_div to compare it with mmd
    opt = optax.adam(learning_rate=0.1)
    opt_state = opt.init(weights)
    
    @jax.jit
    def update_step(params, opt_state):

        (loss_val, px), grads = jax.value_and_grad(qcbm.mmd_loss, has_aux=True)(params)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        kl_div = -jnp.sum(qcbm.py * jnp.nan_to_num(jnp.log(px / qcbm.py)))

        return params, opt_state, loss_val, kl_div


    # Training loop    
    history = []
    divs = []
    n_iterations = 100
    
    for i in range(n_iterations):
        
        weights, opt_state, loss_val, kl_div = update_step(weights, opt_state)

        if i % 10 == 0:
            print(f"Step: {i} Loss: {loss_val:.4f} KL-div: {kl_div:.4f}")

        history.append(loss_val)
        divs.append(kl_div)

    
    # Plotting mmd and kl divergence
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(history)
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("MMD Loss")
    
    ax[1].plot(divs, color="green")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("KL Divergence")
    plt.show()

    
if __name__ == "__main__":
    main()
