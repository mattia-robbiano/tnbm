import sys
import quimb
import quimb.tensor as qtn
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('..')
from functions import *

"""dataset of measurements in computational basis of a 2-qubit system
|T|=4
"""
n = 9
dataset = get_bars_and_stripes(int(np.sqrt(n)))

"""create a hyperindexed representation of training set as a quimb tensor network
keep in mind that [0,1] means measuring |0> and |1> on the two qubits. The elements of computational basis
can be represented by vectors (0,1) and (1,0) respectively.
The training set is represented by an MPO. Each node represent all the qubit measurement as a matrix.
Each row of the matrix is single mesurements of the qubit in computational basis (0,1) or (1,0)
We will have as many rows as the number of data points in the training set.

In our example we will have 2 tensors, each with 4 rows and 2 columns.
"""
measurements = [[data[i] for data in dataset] for i in range(n)]
training_tensor_data = [np.array([[1, 0] if m == 0 else [0, 1] for m in meas]) for meas in measurements]
training_tensors = [qtn.Tensor(data=training_tensor_data[i], inds=('hyper', f'cbase{i}'), tags=f'sample{i}') for i in range(n)]
training_tensor_network = qtn.TensorNetwork(training_tensors)

""" Initializing psi as MPS to be trained
"""
psi = qtn.MPS_rand_state(n, bond_dim=8)
#rename_dict = {f'k{i}': f'cbase{i}' for i in range(n)}
#psi.reindex_(rename_dict)
for i, tensor in enumerate(psi):
    tensor.add_tag(f'psi{i}')

""" Define kernel MPO, all tensors are 2x2 matrices node_matrix
"""
sigma = 0.09
node_matrix = np.array([[1, np.exp(-(1/(2*sigma**2)))], [np.exp(-(1/(2*sigma**2))), 1]])
kernel_tensors = [qtn.Tensor(data=node_matrix, inds=(f'cbase{i}', f'k{i}'), tags=f'kernel{i}') for i in range(n)]
kernel = qtn.TensorNetwork(kernel_tensors)

""" Full model
"""
model = training_tensor_network & kernel & psi

# region drawing
"""Drawing part

1. Place training MPO nodes (with tags 'sample{i}') in a vertical column at x=0.
2. Fix the common 'hyper' index (all left bonds meet here).
3. Place ψ MPS nodes in a vertical column at x=2.
4. Generate a color map
5. Assign same color to each (sample{i}, psi{i}) pair
6. Drawing and saving
"""
import matplotlib.colors as mcolors
import matplotlib.cm as cm

fix_positions = {f'sample{i}': (0, i) for i in range(n)}
fix_positions['hyper'] = (-2, (n - 1) / 2)
fix_positions.update({f'psi{i}': (2, i) for i in range(n)})
fix_positions.update({f'kernel{i}': (1, i) for i in range(n)})

fig = model.draw(
    color = [f'sample{i}' for i in range(n)] + [f'psi{i}' for i in range(n)] + [f'kernel{i}' for i in range(n)],
    fix=fix_positions,
    show_left_inds=True,
    show_tags=True,
    node_size=2,
    figsize=(8, 12),
    return_fig=True
)
fig.savefig('tensor_network_plot.png', bbox_inches='tight')
#endregion drawing

sys.exit()

""" Contracting
"""
contracted = model.contract(output_inds = [], optimize = 'auto-hq')