import jax.numpy as jnp
import quimb.tensor as qtn

def MMD(psi, training_tensor_network, kernel):  

    #TODO reindex hyper in hyper1
    rename_dict = {f'k{i}': f'cbase{i}' for i in range(psi.L)}
    rename_hyper_dict = {'hyper': 'hyper1'}
    psi_copy = psi.reindex(rename_dict)
    training_tensor_network_copy = training_tensor_network.reindex(rename_dict)
    training_tensor_network_copy.reindex_(rename_hyper_dict)

    mix_term = (psi | kernel | training_tensor_network).contract(output_inds = [], optimize = 'auto-hq')
    homogeneous_term_q = (psi | kernel | psi_copy).contract(output_inds = [], optimize = 'auto-hq')
    homogeneous_term_p = (training_tensor_network | kernel | training_tensor_network_copy).contract(output_inds = [], optimize = 'auto-hq')

    loss_value = homogeneous_term_q -2*mix_term  + homogeneous_term_p
    
    return loss_value

def KLD(psi, training_tensor_network, kernel):
    
    loss_value = 0
    rename_dict = {f'k{i}': f'cbase{i}' for i in range(psi.L)}
    psi.reindex_(rename_dict)

    ampl = (psi & training_tensor_network).contract(output_inds = ['hyper'], optimize = 'auto-hq').data
    qx = jnp.abs(ampl)**2
    px = 1/training_tensor_network.ind_size('hyper')
    loss_value = jnp.sum(px*jnp.log(px / qx),0)    
    
    return loss_value



