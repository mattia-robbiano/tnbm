import sys
sys.path.append('../')
import pickle
import numpy as np
import quimb.tensor as qtn
from functions.dataset_utils import bars_and_stripes, hypertn_from_data
from functions.loss import nll_loss, mmd_loss


chi_list =       [2, 4, 8, 16, 32]#, 64] sul cluster
num_sites_list = [2**2, 3**2, 4**2, 5**2, 6**2]#, 7**2] sul cluster

# Building the loss function tensors
result = []
for num_sites in num_sites_list:

    comp_basis_povm = np.array([np.array([[1,0],[0,0]]), np.array([[0,0],[0,1]])])
    tensors = []
    for i in range(num_sites):
        t = qtn.Tensor(comp_basis_povm,inds=(f'o{i}', f'k{i}', f'b{i}'), tags = [f'I{i}', 'POVM'])
        tensors.append(t)
    povm_tn = qtn.TensorNetwork(tensors)
    povm_tn = povm_tn / povm_tn.norm()

    sigma = 0.55
    fact = np.exp(-1. / (2 * sigma**2))
    kernel_mat = np.array([[1, fact],[fact, 1]])
    kernel_mpo = qtn.MPO_product_operator([kernel_mat]*num_sites,upper_ind_id = 'o{}',lower_ind_id = 'op{}',tags = ['KERNEL'])
    kernel_mpo = kernel_mpo / kernel_mpo.norm()

    bas = bars_and_stripes(int(np.sqrt(num_sites)), shuffle=True)
    num_samples = bas.shape[0]
    htn_freq = hypertn_from_data(data=bas)/num_samples

    for chi in chi_list:
        psi_list = [qtn.MPS_rand_state(num_sites, bond_dim=chi, dist='uniform') for _ in range(100)]

        mmd_list = [mmd_loss(psi, povm_tn, kernel_mpo, htn_freq, contraction_method='auto-hq') for psi in psi_list]
        nll_list = [nll_loss(psi, htn_freq) for psi in psi_list]

        mmd_variance = np.var(mmd_list)
        nll_variance = np.var(nll_list)

        mmd_min = np.min(mmd_list)
        nll_min = np.min(nll_list)

        mmd_nlog_norm_var = - np.log(mmd_variance / mmd_min)
        nll_nlog_norm_var = - np.log(nll_variance / nll_min)

        print(f"num_sites: {num_sites}, chi: {chi}, mmd_log_norm_var: {mmd_nlog_norm_var}, nll_log_norm_var: {nll_nlog_norm_var}")

        result.append({'num_sites': num_sites,'bond_dimension': chi,'loss_type': 'mmd', 'nlog_norm_var': mmd_nlog_norm_var,})
        result.append({'num_sites': num_sites,'bond_dimension': chi,'loss_type': 'nll', 'nlog_norm_var': nll_nlog_norm_var,})

    with open('result.pkl', 'wb') as f:
        pickle.dump(result, f)