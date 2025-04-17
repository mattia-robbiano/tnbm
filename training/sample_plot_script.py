import pickle
from plotting_utils import plot_BS

with open('tensor_network.pkl', 'rb') as f:
    psi = pickle.load(f)

plot_BS(tn=psi, 
        loss_function='MMD',
        dataset='BAS',
        num_qubits=16,
        bond_dimension=20,
        iterations=500
        )