import time

import pickle
import quimb
import quimb.tensor as qtn
import numpy as np
import matplotlib.pyplot as plt

seed = int(time.time())
test = "GHZ"    # GHZ state
#test = "BS"     # Bars and Stripes

with open('results/tensor_network.pkl', 'rb') as f:
    tn = pickle.load(f)

if test == "BS":
    num_images = 10
    num_columns = 5 

    fig, axes = plt.subplots(2, num_columns, figsize=(15, 6))

    axes = axes.flatten()

    for i, b in enumerate(tn.sample(num_images, seed)):
        arr = np.array(b[0]).reshape((3, 3))
        axes[i].imshow(arr, cmap='gray', interpolation='nearest')
        axes[i].set_title(f'Image {i+1}')

    plt.tight_layout()
    plt.savefig("results/sampled_images.png")

elif test == "GHZ":
    num_samples = 1000
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
    plt.savefig("results/ghz_histogram.png")