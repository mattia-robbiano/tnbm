import numpy as np
import jax
import jax.numpy as jnp
import quimb
import quimb.tensor as qtn
import jax.random as random
key = random.PRNGKey(0)  # Use any integer seed you like.

key, subkey = random.split(key)  # Ensure you manage your PRNG key properly.


def perfect_sampling(TN):
    n_qubits = 8
    # initialize JAX arrays
    s_hat = jnp.zeros((n_qubits, 2))
    probability = jnp.zeros(n_qubits)

    for i in range(n_qubits):
        stepTN = TN
        for j in range(i): 
            # connecting original TN with all extracted samples (tensors)
            v = qtn.Tensor(
                data=s_hat[j], 
                inds=[f'k{j}'], 
                tags=[f'v{int(s_hat[j][0])}']
            )
            stepTN = (stepTN & v) / jnp.sqrt(probability[j])
        
        # computing marginal
        reducedTN = (stepTN.H.reindex({f'k{i}': f'k{i}*'}) & stepTN).contract(optimize='auto-hq')
        v0 = qtn.Tensor(data=jnp.array([0, 1]), inds=[f'k{i}'])
        v1 = qtn.Tensor(data=jnp.array([1, 0]), inds=[f'k{i}'])
        ps0 = (v0 & reducedTN & v0.reindex({f'k{i}': f'k{i}*'})).contract(optimize='auto-hq')
        ps1 = (v1 & reducedTN & v1.reindex({f'k{i}': f'k{i}*'})).contract(optimize='auto-hq')
    
        # check if the probability is normalized
        # if ps0 + ps1 < 0.999 or ps0 + ps1 > 1.001:
        #     print("Error at cycle:", i)
        #     print("ps0:", ps0)
        #     print("ps1:", ps1)
        
        # extracting new element; here we still use np.random for simplicity
        r = random.uniform(subkey, shape=())  # Generates a scalar JAX array.
        if r < ps0:
            s_hat = s_hat.at[i].set(v0.data)
            probability = probability.at[i].set(ps0)
        else:
            s_hat = s_hat.at[i].set(v1.data)
            probability = probability.at[i].set(ps1)
    
    # post-processing: converting [0, 1] -> 0 and [1, 0] -> 1
    for i in range(n_qubits):
        if jnp.array_equal(s_hat[i], jnp.array([0, 1])):
            s_hat = s_hat.at[i].set(jnp.array([0]))
        elif jnp.array_equal(s_hat[i], jnp.array([1, 0])):
            s_hat = s_hat.at[i].set(jnp.array([1]))
        else:
            print("Error: wrong value")
            raise ValueError
    
    s_hat = s_hat.T
    x = s_hat[0]
    
    return x
