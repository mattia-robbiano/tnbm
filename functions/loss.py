import numpy as np
import jax.numpy as jnp
import quimb.tensor as qtn
import cotengra as ctg

ctg_opt = ctg.ReusableHyperOptimizer(max_time = 20, 
                                     minimize='combo',
                                     slicing_opts={'target_size': 2**40},         
                                     slicing_reconf_opts={'target_size': 2**28},  
                                     reconf_opts={'subtree_size': 14},
                                     parallel=True,
                                     progbar=True,
                                     directory=True)

def mmd_loss(psi, povm_tn, kernel_mpo, htn_data, contraction_method='auto-hq'):
    """
    Compute the Maximum Mean Discrepancy (MMD) loss for quantum tensor networks.
    This function calculates the MMD loss, which measures the discrepancy between 
    two probability distributions represented by tensor networks. The loss is 
    computed using a kernel method and supports different contraction strategies.
    Args:
        psi (qtn.TensorNetwork): The quantum state tensor network.
        povm_tn (qtn.TensorNetwork): The POVM (Positive Operator-Valued Measure) tensor network.
        kernel_mpo (qtn.TensorNetwork): The kernel matrix product operator (MPO) tensor network.
        htn_data (qtn.TensorNetwork): The tensor network representing the data distribution.
        contraction_method (str, optional): The contraction method to use. Options are:
            - 'opt': Use an optimized contraction strategy.
            - 'qubitwise': Contract qubit by qubit.
            - 'auto-hq': Use an automatic high-quality contraction strategy (default).
    Returns:
        float: The computed MMD loss value.
    Notes:
        - The function constructs three tensor networks:
            1. `model_tn`: Represents the model distribution.
            2. `overlap_data_tn`: Represents the overlap between the model and data distributions.
            3. `data_tn`: Represents the data distribution.
        - The loss is computed as:
            `hom1 - 2 * mix + hom2`
          where `hom1`, `mix`, and `hom2` are the contracted results of the respective tensor networks.
        - The contraction strategy can significantly affect performance and accuracy.
    """
    num_sites = psi.L
    probs_tn = qtn.TensorNetwork([psi.H, povm_tn,psi.reindex_sites('b{}')])

    reindex_map = {f'o{i}': f'op{i}' for i in range(num_sites)}
    model_tn = qtn.TensorNetwork([probs_tn, kernel_mpo, probs_tn.reindex(reindex_map)])
    
    reindex_map_op = {f'k{i}': f'op{i}' for i in range(num_sites)}
    overlap_data_tn = qtn.TensorNetwork([probs_tn,kernel_mpo, htn_data.reindex(reindex_map_op)])

    reindex_map_o = {f'k{i}': f'o{i}' for i in range(num_sites)}
    data_tn = qtn.TensorNetwork([htn_data.reindex(reindex_map_o), kernel_mpo, htn_data.reindex(reindex_map_op)])

    if contraction_method == 'opt':
        hom1 = model_tn.contract(output_inds=[], optimize=ctg_opt)
        mix = overlap_data_tn.contract(output_inds=[], optimize=ctg_opt)
        hom2 = data_tn.contract(output_inds=[], optimize=ctg_opt)
        return hom1.real - 2 * mix.real + hom2.real
    
    if contraction_method == 'qubitwise':
        for i in range(psi.L):
            model_tn = model_tn.contract_tags_(f'I{i}', optimize='auto-hq')
            overlap_data_tn = overlap_data_tn.contract_tags_(f'I{i}', optimize='auto-hq')
            data_tn = data_tn.contract_tags_(f'I{i}', optimize='auto-hq')
        
    hom1 = model_tn.contract(output_inds=[], optimize='auto-hq')
    mix = 2 * overlap_data_tn.contract(output_inds=[], optimize='auto-hq')
    hom2 = data_tn.contract(output_inds=[], optimize='auto-hq')
    
    return hom1.real - 2 * mix.real + hom2.real

def nll_loss(psi, htn_data, contraction_method='auto-hq' ):
    """
    Compute the negative log-likelihood (NLL) loss for a given tensor network.
    Parameters:
    -----------
    psi : TensorNetwork
        The tensor network representing the model or state.
    htn_data : TensorNetwork
        The tensor network representing the data or target.
    contraction_method : str, optional
        The method used for tensor contraction. Options are:
        - 'opt': Use an optimized contraction path.
        - 'qubitwise': Perform qubit-wise contraction.
        - 'auto-hq' (default): Use an automatic high-quality contraction method.
    Returns:
    --------
    float
        The negative log-likelihood loss value.
    Notes:
    ------
    - The loss is computed as the negative mean of the logarithm of the squared 
        absolute values of the contracted tensor network.
    - The contraction method determines how the tensor networks are combined 
        and contracted to compute the loss.
    """
    
    if contraction_method == 'opt':
        loss = (psi | htn_data).contract(output_inds = ['hyper'], optimize = ctg_opt).data
    elif contraction_method == 'qubitwise':
        for i in range(psi.L):
            loss_tn = loss_tn.contract_tags_(f'I{i}', optimize='auto-hq')
        loss = loss_tn.contract(output_inds = ['hyper'], optimize = 'auto-hq').data 
    else:
        loss = (psi | htn_data).contract(output_inds = ['hyper'], optimize = 'auto-hq').data
    
    return -1. * jnp.mean(jnp.log(jnp.abs(loss) ** 2))
