import time

import pickle
import quimb
import quimb.tensor as qtn
import numpy as np
import matplotlib.pyplot as plt

seed = int(time.time())

with open('tensor_network.pkl', 'rb') as f:
    tn = pickle.load(f)

num_images = 10
num_columns = 5 

fig, axes = plt.subplots(2, num_columns, figsize=(15, 6))

axes = axes.flatten()

for i, b in enumerate(tn.sample(num_images, seed)):
    arr = np.array(b[0]).reshape((3, 3))
    axes[i].imshow(arr, cmap='gray', interpolation='nearest')
    axes[i].set_title(f'Image {i+1}')

plt.tight_layout()
plt.savefig("sampled_images.png")