import sys
import pickle
import numpy as np
import quimb.tensor as qtn

sys.path.append('../')
from functions.dataset_utils import bars_and_stripes, hypertn_from_data
from functions.loss import mmd_loss, nll_loss
from functions.plotting_utils import plot_BS

"""
Initialize the state to be optimized:
- psi: state to be optimized, inizialized as MPS with random state in with complex dtype
- chi: bond dimension of the MPS
- num_sites: number of sites in the MPS
"""
num_sites = 9
chi = 32
dtype = np.float64
psi = qtn.MPS_rand_state(num_sites, chi, dtype=dtype, tags = ['PSI'])

"""
Create MPO representation of the POVM.
p0 and p1 are the POVM projectors over computational base (onehot encoded).
Each node of the MPO is a tensor with 3 indices:
Indices o{} are used to label the output of the tensor (in this case 0 and 1 for each qubit)
k{} and b{} will be contracted with psi and psi.H 
"""
p0 = np.array([[1,0],[0,0]]) 
p1 = np.array([[0,0],[0,1]])
comp_basis_povm = np.array([p0, p1])
tensors = []
for i in range(num_sites):
    t = qtn.Tensor(comp_basis_povm,inds=(f'o{i}', f'k{i}', f'b{i}'), tags = [f'I{i}', 'POVM'])
    tensors.append(t)
povm_tn = qtn.TensorNetwork(tensors)
povm_tn = povm_tn / povm_tn.norm()

"""
Building the kernel matrix product operator (MPO) for the MMD loss function.
Each node of the MPO is a rank-2 tensor (matrix kernel_mat for gaussian kernel)
Index o{} and op{} is used to label the output of the tensor (in this case 0 and 1 for each qubit)
"""
sigma = 0.55
fact = np.exp(-1. / (2 * sigma**2))
kernel_mat = np.array([[1, fact],[fact, 1]])
kernel_mpo = qtn.MPO_product_operator([kernel_mat]*num_sites, 
                                      upper_ind_id = 'o{}', 
                                      lower_ind_id = 'op{}', 
                                      tags = ['KERNEL'])
kernel_mpo = kernel_mpo / kernel_mpo.norm()

"""
Create the dataset for the training, in form of a list of bitstrings. This dataset is then converted into a hyper tensor network by hypertn_from_data function, where each value of the hyperindex labels a different datapoint.
"""
bas = bars_and_stripes(int(np.sqrt(num_sites)), shuffle=True)
num_samples = bas.shape[0]
htn_freq = hypertn_from_data(data=bas)/num_samples

"""
Optimization through TNOptimizer class of quimb.tensor.
"""
tnopt = qtn.TNOptimizer(
                    tn = psi,
                    loss_fn=nll_loss,
                    # loss_constants={"povm_tn":povm_tn, "kernel_mpo":kernel_mpo, "htn_data":htn_freq},
                    loss_constants={"htn_data":htn_freq},
                    loss_kwargs={'contraction_method':'auto-hq'},
                    norm_fn=lambda x: x / x.norm(),
                    autodiff_backend='jax',
                    jit_fn=True,
                    optimizer='adam',
                    )
iterations = 1000
psi_opt = tnopt.optimize(iterations)

# Plotting the results
fig, ax = tnopt.plot()
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
#fig.savefig('plot.pdf', facecolor='white')      
# Save the optimized tensor network to a file for later use
with open('tensor_network.pkl', 'wb') as f: pickle.dump(psi_opt, f)
# Sampling from the optimized tensor network and plotting the results
plot_BS(tn= psi_opt,loss_function="nnl", dataset="Bars and Stripes", num_qubits=num_sites, bond_dimension=chi, iterations=iterations)