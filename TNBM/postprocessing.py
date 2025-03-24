import time
import re
import pickle
import quimb
import quimb.tensor as qtn
import numpy as np
import matplotlib.pyplot as plt
from main import loss_fn, loss_mpo_builder, dataset_mps_builder


def plot_BS(PATH, num_images=10, num_columns = 5):
    
    
    seed = int(time.time())

    # Extract model information from the PATH
    filename = PATH.split('/')[-1]
    parts = filename.split('_')
    loss_function = parts[0]
    dataset = parts[1]
    num_qubits = float(parts[2].replace('q', ''))
    bond_dimension = int(parts[3].replace('b.pkl', ''))

    with open(PATH, 'rb') as f:
        tn = pickle.load(f) 
    fig, axes = plt.subplots(2, num_columns, figsize=(15, 6))
    axes = axes.flatten()

    for i, b in enumerate(tn.sample(num_images, seed)):
        arr = np.array(b[0]).reshape((int(np.sqrt(num_qubits)), int(np.sqrt(num_qubits))))
        axes[i].imshow(arr, cmap='gray', interpolation='nearest')
        axes[i].set_title(f'Image {i+1}')

    plt.suptitle(f'Samples - {loss_function} - {dataset} - {num_qubits} qubits - Bond dim {bond_dimension}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the title
    plt.savefig("samples"+ "_" +loss_function + "_" + dataset + "_" + str(num_qubits) + "q_" + str(bond_dimension) + "b.png")

def plot_histogram(num_samples=1000):

    seed = int(time.time())
    with open('tensor_network.pkl', 'rb') as f:
        tn = pickle.load(f)

    samples = tn.sample(num_samples, seed)  # Get all samples at once

    # Convert bitstrings to string representation
    bitstrings = [''.join(map(str, sample[0])) for sample in samples]

    # Compute histogram data
    unique, counts = np.unique(bitstrings, return_counts=True)

    # Ensure all possible bitstrings of the given length are on the x-axis
    bitstring_length = len(bitstrings[0])
    all_bitstrings = [format(i, f'0{bitstring_length}b') for i in range(2**bitstring_length)]
    all_counts = [counts[unique.tolist().index(bs)] if bs in unique else 0 for bs in all_bitstrings]

    # Plot histogram
    plt.bar(all_bitstrings, all_counts)
    plt.xlabel('Bitstrings')
    plt.ylabel('Counts')
    plt.title('Histogram of GHZ State Bitstrings')
    plt.xticks([])    
    plt.tight_layout()
    plt.savefig("ghz_histogram.png")

def extract_first_unsigned_number(s):
    match = re.search(r"\d*\.\d+|\d+", s)
    return float(match.group(0)) if match else None 

def plot_loss(file1, file2):
    

    # Extract model information from the file paths
    def extract_info(file_path):
        filename = file_path.split('/')[-1]
        parts = filename.split('_')
        loss_function = parts[0]
        dataset = parts[1]
        num_qubits = float(parts[2].replace('q', ''))
        bond_dimension = int(parts[3].replace('b.out', ''))
        return loss_function, dataset, num_qubits, bond_dimension

    loss_function1, dataset1, num_qubits1, bond_dimension1 = extract_info(file1)
    loss_function2, dataset2, num_qubits2, bond_dimension2 = extract_info(file2)

    """Reads two files, extracts numbers from each line, and plots them for comparison."""
    numbers1, numbers2 = [], []
    max_lines = 1000

    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        # Skip first 2 lines
        for _ in range(2):
            next(f1)
            next(f2)
        
        for line1, line2 in zip(f1, f2):
            if len(numbers1) >= max_lines:
                break
            num1 = extract_first_unsigned_number(line1)
            num2 = extract_first_unsigned_number(line2)

            if num1 is not None and num2 is not None:
                numbers1.append(num1)
                numbers2.append(num2)
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(numbers1, label=file1, linestyle="-", linewidth=1, marker="")
    if file1 != file2:
        plt.plot(numbers2, label=file2, linestyle="-", linewidth=1, marker="")    
    plt.xlabel("epochs")
    plt.ylabel("MMD")
    plt.yscale('log')  # Set y-axis to log scale
    if file1 == file2:
        plt.title(f"Loss - {loss_function1} - {dataset1} - {num_qubits1} qubits - Bond dim {bond_dimension1}")
    else:
        plt.title(f"Loss - {loss_function1} - {dataset1} - {num_qubits1} qubits - Bond dim {bond_dimension1} vs {loss_function2} - {dataset2} - {num_qubits2} qubits - Bond dim {bond_dimension2}")
    plt.legend()
    plt.grid()
    if file1 == file2:
        plt.savefig(f"loss_{loss_function1}_{dataset1}_{num_qubits1}q_{bond_dimension1}b.png")
    else:
        plt.savefig(f"loss_{loss_function1}_{dataset1}_{num_qubits1}q_{bond_dimension1}b_vs_{loss_function2}_{dataset2}_{num_qubits2}q_{bond_dimension2}b.png")

def plot_variance(filename, bond_dims=None):
    """
    Plot variance vs. n for selected bond dimensions.

    :param filename: File containing variance data.
    :param bond_dims: List of bond dimensions to plot (None = plot all).
    """
    data = {}

    # Read and parse the file
    with open(filename, 'r') as f:
        for line in f:
            match = re.search(r'Variance for n = (\d+) and bond dimension (\d+) is ([\deE.-]+)', line)
            if match:
                n, b, variance = int(match.group(1)), int(match.group(2)), float(match.group(3))
                if b not in data:
                    data[b] = []
                data[b].append((n, variance))

    # Sort data for proper plotting
    for b in data:
        data[b].sort()  # Sort by n

    # Select bond dimensions to plot
    selected_bonds = bond_dims if bond_dims else sorted(data.keys())

    # Plot
    plt.figure(figsize=(8, 6))
    for b in selected_bonds:
        if b in data:
            ns, variances = zip(*data[b])
            plt.plot(ns, variances, marker='o', linestyle='-', label=f'Bond dim {b}')

    plt.xscale('linear')  # Keep x-axis linear
    plt.yscale('log')  # Log scale for variance
    plt.xlabel('n')
    plt.ylabel('MMD Variance')
    plt.legend()
    plt.title('Loss concentration - MMD - Bars and Stripes')
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xticks(range(min(ns), max(ns) + 1))
    plt.savefig("variance_plot.png")

def loss(tn = None):
    if tn is None:
        with open('tensor_network.pkl', 'rb') as f:
            tn = pickle.load(f)
    mpo = loss_mpo_builder(loss= "mmd", sigma = 0.09, dimension=9)
    data = dataset_mps_builder(dimension=9, default_dataset="BS")
    loss = loss_fn(psi=tn, training_tensor_network=data, kernel=mpo)
    return loss 

    