import time
import pickle
import numpy as np
import quimb.tensor as qtn
import cotengra as ctg
import matplotlib.pyplot as plt
from dataset_utils import bars_and_stripes, hypertn_from_data

num_sites = 16
chi = 20

psi = qtn.MPS_rand_state(num_sites, chi, dtype=np.complex128, tags = ['PSI'])
# Projectors
p0 = np.array([[1,0],[0,0]]) 
p1 = np.array([[0,0],[0,1]])
comp_basis_povm = np.array([p0, p1])

# Create MPO representation of the POVM.
# Indices o{} are used to label the possible outcomes (in this case 0 and 1 for each qubit)
tensors = []
for i in range(num_sites):
    t = qtn.Tensor(comp_basis_povm,
                   inds=(f'o{i}', f'k{i}', f'b{i}'), 
                   tags = [f'I{i}', 'POVM'])
    tensors.append(t)

povm_tn = qtn.TensorNetwork(tensors)
probs_tn = qtn.TensorNetwork([psi.H, povm_tn, psi.reindex_sites('b{}')])

sigma = 0.5
fact = 1/(2*sigma**2)
kernel_mat = np.array([[1, fact],[fact, 1]])

# Kernel MPO
kernel_mpo = qtn.MPO_product_operator([kernel_mat]*num_sites, 
                                      upper_ind_id = 'o{}', 
                                      lower_ind_id = 'op{}', 
                                      tags = ['KERNEL'])
kernel_mpo = kernel_mpo / kernel_mpo.norm()


reindex_map = {f'o{i}': f'op{i}' for i in range(num_sites)}
overlap_tn = qtn.TensorNetwork([probs_tn, 
                                kernel_mpo,
                                probs_tn.reindex(reindex_map)])


bas = bars_and_stripes(int(np.sqrt(num_sites)), shuffle=False)
htn_data = hypertn_from_data(bas)

reindex_map = {f'k{i}': f'op{i}' for i in range(num_sites)}
overlap_data_tn = qtn.TensorNetwork([probs_tn, 
                                kernel_mpo,
                                htn_data.reindex(reindex_map) ])

ctg_opt = ctg.ReusableHyperOptimizer(max_time = 20, 
                                     minimize='combo',
                                     slicing_opts={'target_size': 2**40},         
                                     slicing_reconf_opts={'target_size': 2**28},  
                                     reconf_opts={'subtree_size': 14},
                                     parallel=True,
                                     progbar=True,
                                     directory=True)

def mmd_loss(psi, povm_tn, kernel_mpo, htn_data, contraction_method='auto-hq'):
    
    probs_tn = qtn.TensorNetwork([psi, povm_tn,psi.H.reindex_sites('b{}')])

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
    
    if contraction_method == 'opt':
        loss = (psi | htn_data).contract(output_inds = ['hyper'], optimize = ctg_opt).data
    elif contraction_method == 'qubitwise':
        for i in range(psi.L):
            loss_tn = loss_tn.contract_tags_(f'I{i}', optimize='auto-hq')
        loss = loss_tn.contract(output_inds = ['hyper'], optimize = 'auto-hq').data 
    else:
        loss = (psi | htn_data).contract(output_inds = ['hyper'], optimize = 'auto-hq').data
    
    return -1. * np.mean(np.log(np.abs(loss) ** 2))

tnopt = qtn.TNOptimizer(
                    tn = psi,
                    loss_fn=mmd_loss,
                    loss_constants={"povm_tn":povm_tn, "kernel_mpo":kernel_mpo, "htn_data":htn_data},
                    # loss_fn=nnl_loss,
                    # loss_constants={"htn_data":htn_data},
                    loss_kwargs={'contraction_method':'opt'},
                    norm_fn=lambda x: x / x.norm(),
                    autodiff_backend='jax',
                    jit_fn=True,
                    optimizer='adam',
                    )
iterations = 1000
psi_opt = tnopt.optimize(iterations)

fig, ax = tnopt.plot()
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
fig.savefig('plot.pdf', facecolor='white')      

with open('tensor_network.pkl', 'wb') as f: pickle.dump(psi_opt, f)


def plot_BS(loss_function, dataset, num_qubits, bond_dimension, iterations, num_images=20, num_columns = 10):
    
    seed = int(time.time())
    tn = psi_opt
    fig, axes = plt.subplots(2, num_columns, figsize=(15, 6))
    axes = axes.flatten()

    for i, b in enumerate(tn.sample(num_images, seed)):
        arr = np.array(b[0]).reshape((int(np.sqrt(num_qubits)), int(np.sqrt(num_qubits))))
        axes[i].imshow(arr, cmap='gray', interpolation='nearest',vmin = 0, # for binary images
                       vmax = 1)
        axes[i].set_title(f'Image {i+1}')

    plt.suptitle(f'Samples - {loss_function} - {dataset} - {num_qubits} qubits - Bond dim {bond_dimension} - Iterations {iterations}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    plt.savefig("samples"+ "_" +loss_function + "_" + dataset + "_" + str(num_qubits) + "q_" + str(bond_dimension) + "b_" + str(iterations) + "i.pdf")
    
plot_BS("MMD", "Bars and Stripes", num_qubits=num_sites, bond_dimension=chi, iterations=iterations)