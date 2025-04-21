import sys
sys.path.append('../')
import numpy as np
import quimb.tensor as qtn

import matplotlib.pyplot as plt
#import scienceplots
from functions.dataset_utils import bars_and_stripes, hypertn_from_data
from functions.loss import nll_loss, mmd_loss
#plt.style.use("science")

# parameters
chi_list = [2,10,30,50]
num_sites_list = [2**2, 3**2, 4**2, 5**2, 6**2, 7**2, 8**2, 9**2]
use_mmd_loss = True

# computing variance
variance_list = []
for num_sites in num_sites_list:
    print(f"Number of sites: {num_sites}")

    p0 = np.array([[1,0],[0,0]]) 
    p1 = np.array([[0,0],[0,1]])
    comp_basis_povm = np.array([p0, p1])
    tensors = []
    for i in range(num_sites):
        t = qtn.Tensor(comp_basis_povm,inds=(f'o{i}', f'k{i}', f'b{i}'), tags = [f'I{i}', 'POVM'])
        tensors.append(t)
    povm_tn = qtn.TensorNetwork(tensors)
    povm_tn = povm_tn / povm_tn.norm()

    sigma = 0.55
    fact = np.exp(-1. / (2 * sigma**2))
    kernel_mat = np.array([[1, fact],[fact, 1]])
    kernel_mpo = qtn.MPO_product_operator([kernel_mat]*num_sites, 
                                        upper_ind_id = 'o{}', 
                                        lower_ind_id = 'op{}', 
                                        tags = ['KERNEL'])
    kernel_mpo = kernel_mpo / kernel_mpo.norm()

    bas = bars_and_stripes(int(np.sqrt(num_sites)), shuffle=True)
    num_samples = bas.shape[0]
    htn_freq = hypertn_from_data(data=bas)/num_samples
    
    variance_chi_list = []
    for chi in chi_list:
        loss_list = []
        contraction_method = 'opt' if num_sites > 25 or (num_sites == 25 and chi > 20) else 'auto-hq'
        for _ in range(100):
            psi = qtn.MPS_rand_state(num_sites, bond_dim=chi, dist='uniform')
            if use_mmd_loss:
                loss = mmd_loss(psi, povm_tn, kernel_mpo, htn_freq, contraction_method=contraction_method)
            else:
                loss = nll_loss(psi, htn_freq)
            loss_list.append(loss)
        variance= np.var(loss_list)
        print(variance)
        variance_chi_list.append(variance)
    variance_list.append(variance_chi_list)

# plotting variance_list
plt.figure(figsize=(6, 6*6/8))
for j, chi in enumerate(chi_list):
    variance_for_chi = [variance_list[i][j] for i in range(len(num_sites_list))]
    plt.plot(num_sites_list, variance_for_chi, marker='*', label=f'Bond Dim {chi}')
plt.yscale('log')

if use_mmd_loss == True: plt.title('Loss concentration MMD', fontsize=12)
else: plt.title('Loss concentration MMD', fontsize=12)
plt.xlabel('Number of Qubits', fontsize=12)
plt.ylabel('Variance (Log Scale)', fontsize=12)
plt.xticks(num_sites_list, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(visible=False)
plt.legend(fontsize=10, title_fontsize=10, ncol=1, bbox_to_anchor=(0.83, 0.87), loc='upper center')
plt.tight_layout(pad=1.0)
if use_mmd_loss == True:plt.savefig('loss_concentration_MMD.pdf')
else: plt.savefig('loss_concentration_NLL.pdf')
plt.show()