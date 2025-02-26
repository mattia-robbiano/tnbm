"""
DA FARE: 
Ricorda! E' un modello implicito, quindi non conosciamo la distribuzione target (nell'esempio toy facciamo finta di non conoscerla)
Bisogna aggiustare la MMD. Al momento è come quella di di Oriel con exact True, ci serve l'altra, cioè quella per i sample.
Il training non si può fare con autograd così com'è. Bisogna fare un training loop e fare i gradienti con parameter shift.
Bisogna capire come si fa a rendere la bond dimension una roba addestrabile
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

from functions import *


PATH = "parameters.json"
SAMPLE_BITSTRING_DIMENSION, PRINT_TARGET_PDF, DEVICE, EPOCHS = load_parameters(PATH)
jax.config.update("jax_platform_name", DEVICE)
np.random.seed(42)
n_qubits = SAMPLE_BITSTRING_DIMENSION


def main():
    """
    Building dataset. We will then provide the model one data at the time and calculate MMD for each sample
    """
    dataset = get_bars_and_stripes(int(math.sqrt(SAMPLE_BITSTRING_DIMENSION)))
    if PRINT_TARGET_PDF == True:
        print_bitstring_distribution(dataset)


    """
    MPS initialization. The number of open indexes controls the number of nodes in the MPS, and thus the number of 
    gate blocks. Bond_dimension represent maximum bond dimension, that controls the expressivity of the model.
    bond_dimension = 2^(D/2) where N is the number of qubits of the largest state exactly representable.
    TO DO: bond dimension should be some kind of variable
    """
    number_open_index = 9
    bond_dimension =   2
    psi = qtn.MPS_rand_state(number_open_index, bond_dimension)

    """
     OPTIMIZATION
     We have to understand which are our parameters. We have to give a fitting format to data to be given to tn.
     At every loop of r we change learning rate for convergence.
     At every epoch we wanto to compute gradients to update the MPS. We calculate the MMD loss inside the gradient computation.
    """
    sigma = 0.09
    batch_size = 12
    tolerance = 1e-6
    
    # # test
    # random_bitstring = np.array([0,0,0,0,0,0,0,0,0])
    # y = qtn.MPS_computational_state(random_bitstring)
    # for i in range(10):
    #     x = qtn.MPS_rand_state(number_open_index, bond_dimension)
    #     loss = MMD(x, y, sigma, number_open_index, bond_dimension)
    #     print(loss)

    # return 0

    for ep in range(EPOCHS):

        """LOADING SAMPLE"""
        np.random.shuffle(dataset)
        target_train = dataset[:batch_size,...]
        sample_generator = psi.sample(C=batch_size,seed=1234567)  
        samples = np.array([bits for bits, _ in sample_generator])


        """COMPUTING LOSS FUNCTION FOR LOG"""
        loss =  MMD(samples,target_train,sigma)
        loss = float(np.mean(loss))


        """COMPUTING GRADIENTS"""
        # for _ in range(grad_batch):
        #   gradients = compute_gradient(ansatz, parameters, target_train, qubits, n_shots,grad_loss_function, signal = signal, exact = exact, values = values)
        #   median_grad.append(gradients)
        # temp_grad = np.mean(np.mean(np.array(median_grad), axis=0), axis=-1).reshape(-1)
        
        # Dummy tensors update for testing parameters update
        ###############################################
        ghz = qtn.MPS_ghz_state(L=number_open_index) ##
        new_tensor_array = []                        ##
        for site, tensor in ghz.tensor_map.items():  ##
            new_tensor_array.append(tensor.data)     ##
        ###############################################


        """CHECKING CONVERGENCE AND PRINTING"""
        distance = np.sqrt(sum(np.linalg.norm(new - old)**2 for new, old in zip(new_tensor_array, tensor_array)))
        #if np.linalg.norm(np.asarray(new_tensor_array) - np.asarray(tensor_array)) < tolerance:
        if distance < tolerance:
            print(f'Iteration {ep + 1}: cost = {loss:.6f}')
            print(f'Converged in {ep + 1} iterations.')
            return psi


        """PARAMETERS UPDATE"""
        for site, new_tensor in enumerate(new_tensor_array):
            psi.tensor_map[site].data[:] = new_tensor
        
        
        #if (ep + 1) % 100 == 0:
        print(f'Iteration {ep + 1}: cost = {loss:.6f}') #, ||grad|| = {np.linalg.norm(grad):.6f}')

    print("Maximum iterations reached.")
    return psi


if __name__ == "__main__":
    main()