import sys
import pickle
sys.path.append('../')
from functions.plotting_utils import plot_BS

with open('tensor_network.pkl', 'rb') as f:
    psi = pickle.load(f)

plot_BS(tn=psi, 
        loss_function='MMD',
        dataset='BAS',
        num_qubits=9,
        bond_dimension=20,
        iterations=4000
        )