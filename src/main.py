import pickle
import numpy as np
import quimb.tensor as qtn

import jax
from jax import numpy as jnp
jax.config.update("jax_enable_x64", True)

from dataset_utils import bars_and_stripes, hypertn_from_data, plot_binary_data

loss_list = []
def callback_fn(tnopt):
    g = jax.grad(tnopt.loss_fn)(tnopt.get_tn_opt(), **tnopt.loss_constants)
    print("Gradient:", g)

def nnl_loss(tn, htn_data):
    """
    Calculate the KL divergence between the MPS and the dataset.
    """
    loss = (tn | htn_data).contract(output_inds = ['hyper'], 
                                     optimize = 'auto-hq').data
    
    return -1. * jnp.mean( jnp.log( jnp.abs(loss)**2 ) )

def mmd_loss(tn, povm_tn, kernel_mpo, htn_data):
    
    num_sites = tn.L

    probs_tn = qtn.TensorNetwork([tn.H,
                              povm_tn, 
                              tn.reindex_sites('b{}')])
    
    reindex_map = {f'o{i}': f'op{i}' for i in range(num_sites)}
    model_tn = qtn.TensorNetwork([probs_tn, 
                                    kernel_mpo,
                                    probs_tn.reindex(reindex_map)])
    
    reindex_map_op = {f'k{i}': f'op{i}' for i in range(num_sites)}
    overlap_data_tn = qtn.TensorNetwork([probs_tn, 
                                    kernel_mpo,
                                    htn_data.reindex(reindex_map) ])
    
    reindex_map_o = {f'k{i}': f'o{i}' for i in range(num_sites)}
    data_tn = qtn.TensorNetwork([htn_data.reindex(reindex_map_o), 
                                    kernel_mpo,
                                    htn_data.reindex(reindex_map_op)])
    
    mmd = model_tn.contract(output_inds=[], optimize='auto-hq') - 2*overlap_data_tn.contract(output_inds=[], optimize='auto-hq') + data_tn.contract(output_inds=[], optimize='auto-hq')
    
    return mmd.real


def main():

    ######################## MODEL ########################
    num_sites = 4
    chi = 5
    psi = qtn.MPS_rand_state(num_sites, chi, dtype=np.complex128, tags = ['PSI'])

    ######################## POVM ########################
    # construcing projectors |0X0| and |1X1| and building the POVM
    p0 = np.array([[1,0],[0,0]]) 
    p1 = np.array([[0,0],[0,1]])
    comp_basis_povm = np.array([p0, p1])
    # create MPO representation of the POVM.
    # indices o{} are used to label the possible outcomes (in this case 0 and 1 for each qubit)
    tensors = []
    for i in range(num_sites):
        t = qtn.Tensor(comp_basis_povm,
                    inds=(f'o{i}', f'k{i}', f'b{i}'), 
                    tags = [f'I{i}', 'POVM'])
        tensors.append(t)
    povm_tn =  qtn.TensorNetwork(tensors)

    ######################## KERNEL ########################
    sigma = 0.5
    fact = 1/(2*sigma**2)
    kernel_mat = np.array([[1, fact],[fact, 1]])

    kernel_mpo = qtn.MPO_product_operator([kernel_mat]*num_sites,
                                        upper_ind_id = 'o{}', 
                                        lower_ind_id = 'op{}', # op stands for o prime, o'
                                        tags = ['KERNEL'])

    ######################## DATA ########################
    bas = bars_and_stripes(int(np.sqrt(num_sites)))
    htn_data = hypertn_from_data(bas)

    ######################## NORMALIZATION DEBUG ########################
    kernel_mpo = kernel_mpo / kernel_mpo.norm()

    ######################## OPTIMIZER ########################
    tnopt = qtn.TNOptimizer(
                        tn = psi, 
                        loss_fn=nnl_loss,
                        loss_constants={"htn_data":htn_data},
                        #loss_constants={"povm_tn":povm_tn, "kernel_mpo":kernel_mpo, "htn_data":htn_data},
                        loss_kwargs={},
                        norm_fn=lambda x: x / x.norm(),
                        autodiff_backend='jax',
                        jit_fn=True,
                        optimizer='adam',
                        callback=callback_fn,
                        )
                    
    psi_opt = tnopt.optimize(10)

    ######################## POSTPROCESSING ########################
    # num_samples = 25
    # samples = []
    # for b, p in psi_opt.sample(num_samples):
    #     print(p)
    #     samples.append(b)
    # samples = np.array(samples).reshape(num_samples, int(np.sqrt(num_sites)), int(np.sqrt(num_sites)))
    # plot_binary_data(samples)

    fig, ax = tnopt.plot()
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    fig.savefig('plot.pdf', facecolor='white')
    with open('tensor_network.pkl', 'wb') as f: pickle.dump(psi_opt, f)

if __name__ == "__main__":
    main()