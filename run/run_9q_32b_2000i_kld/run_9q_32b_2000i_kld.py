import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from run import run_training
from metrics import MMD, KLD
from postprocessing import plot_BS

# Set the parameters for the run
iterations = 2000
bond_dimension = 32
dataset = "BS"
num_qubits = 9
loss_function = "kld" # KLD or MMD, !MUST BE CHANGED MANUALLY ALSO IN THE RUN FUNCTION!
metrics = ["mmd", "fidelity"]

run_training(
    bond_dimension=bond_dimension,
    loss_fn=KLD,
    dataset_mode=(dataset, num_qubits),
    epochs=iterations,
    callback_metrics=metrics
    )

plot_BS("/data/mrobbian/IC-Deq-BM/run/run_9q_32b_2000i_kld/tensor_network.pkl",
        loss_function=loss_function,
        dataset=dataset,
        num_qubits=num_qubits,      
        bond_dimension=bond_dimension,
        iterations=iterations,        
        )