import time

import pickle
import quimb
import quimb.tensor as qtn
import numpy as np
import matplotlib.pyplot as plt


def plot_BS(num_images=10, num_columns = 5):

    seed = int(time.time())

    with open('tensor_network.pkl', 'rb') as f:
        tn = pickle.load(f) 

    fig, axes = plt.subplots(2, num_columns, figsize=(15, 6))

    axes = axes.flatten()

    for i, b in enumerate(tn.sample(num_images, seed)):
        arr = np.array(b[0]).reshape((3, 3))
        axes[i].imshow(arr, cmap='gray', interpolation='nearest')
        axes[i].set_title(f'Image {i+1}')

    plt.tight_layout()
    plt.savefig("BS_sampled_images.png")


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

import re

def extract_first_unsigned_number(s):
    match = re.search(r"\d*\.\d+|\d+", s)
    return float(match.group(0)) if match else None 

def plot_numbers_from_files(file1, file2):
    """Reads two files, extracts numbers from each line, and plots them for comparison."""
    numbers1, numbers2 = [], []
    max_lines = 1000

    with open(file1, 'r') as f1, open(file2, 'r') as f2:
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
    plt.plot(numbers1, label="MMD", linestyle="-", linewidth=1, marker="")
    plt.plot(numbers2, label="LQF", linestyle="-", linewidth=1, marker="")    
    plt.xlabel("Line Number")
    plt.ylabel("Extracted Value")
    plt.title("Loss plot - MMD vs LQF")
    plt.legend()
    plt.grid()
    plt.savefig("loss_plot.png")

import matplotlib.pyplot as plt
import numpy as np
import re

import matplotlib.pyplot as plt
import numpy as np
import re

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
    plt.title('Loss concentration - MMD - Cardinality Test')
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xticks(range(min(ns), max(ns) + 1))
    plt.savefig("variance_plot.png")