"""
OBSERVATIONS:
- Right now jax is giving a warning about long time compilation.
- Question: in the paper, the authors mention that the MMD is calculated by tracing the MPO*(psixsample), however in my case that state is an MPS. How do I calculate the trace then?
  At the moment I am computing state @ state.H.
- Execution 1000 iteration ~ 3min on node.
"""
import sys
import json
import numpy as np
import math
import quimb as qu
import jax
import jax.numpy as jnp
import quimb as qu
import quimb.tensor as qtn
import pickle


from functions import *


PATH = "parameters.json"
SAMPLE_BITSTRING_DIMENSION, PRINT_TARGET_PDF, DEVICE, EPOCHS = load_parameters(PATH)
np.random.seed(42)
n_qubits = SAMPLE_BITSTRING_DIMENSION


def main():
    """
    Building dataset. We will then provide the model one data at the time and calculate MMD for each sample.
    The dataset must be composed by bitstrings of the same dimension, that have to be converted in MPS. The MPS list will be passed to the optimizer.
    """
    dataset = get_bars_and_stripes(int(math.sqrt(SAMPLE_BITSTRING_DIMENSION)))
    if PRINT_TARGET_PDF == True:
        print_bitstring_distribution(dataset)

    MPS_dataset = []
    for data in dataset:
        MPS_dataset.append(qtn.MPS_computational_state(data))


    """
    MPS initialization. The number of open indexes controls the number of nodes in the MPS, and thus the number of 
    gate blocks. Bond_dimension represent maximum bond dimension, that controls the expressivity of the model.
    bond_dimension = 2^(D/2) where N is the number of qubits of the largest state exactly representable.

    """
    bond_dimension =   2
    psi = qtn.MPS_rand_state(SAMPLE_BITSTRING_DIMENSION, bond_dim=bond_dimension)


    """
     OPTIMIZATION
     We have to understand which are our parameters. We have to give a fitting format to data to be given to tn.
     At every loop of r we change learning rate for convergence.
     At every epoch we wanto to compute gradients to update the MPS. We calculate the MMD loss inside the gradient computation.

     Ommd initialization. We calculate the MPO operator (more info in the google doc)

    """
    sigma = 0.09
    batch_size = 12

    mpo = Ommd(SAMPLE_BITSTRING_DIMENSION, sigma)


    def loss_fn(psi,dataset,mpo):
        loss = 0
        for data in dataset:
            #y = qtn.MPS_computational_state(data)
            loss += MMD(psi, data, mpo, sigma, SAMPLE_BITSTRING_DIMENSION, bond_dimension)
        loss = loss / len(dataset)

        return loss
    

    tnopt = qtn.TNOptimizer(
        # the tensor network we want to optimize
        psi,

        # the functions specfying the loss and normalization
        loss_fn=loss_fn,

        # we specify constants so that the arguments can be converted
        # to the  desired autodiff backend automatically
        loss_constants={"dataset": MPS_dataset, "mpo": mpo},

        # the underlying algorithm to use for the optimization
        # 'l-bfgs-b' is the default and often good for fast initial progress
        optimizer="adam",

        # which gradient computation backend to use
        autodiff_backend="jax",
    )

    psi_opt = tnopt.optimize(1000)
    fig, ax = tnopt.plot()
    fig.savefig("plot.png")

    """
    Save the tensor network
    """
    with open('tensor_network.pkl', 'wb') as f:
        pickle.dump(psi_opt, f)


    return 0


if __name__ == "__main__":
    main()