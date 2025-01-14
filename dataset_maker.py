import numpy as np
import os

# Create output directories
os.makedirs("imagesA", exist_ok=True)
os.makedirs("imagesB", exist_ok=True)

# Generate and save 100 "outer zero" and 100 "inner zero" matrices
for i in range(100):
    # Outer zero: border is zero, inner pixels normalized to 1
    outer_zero = np.zeros((4, 4), dtype=np.float32)
    outer_zero[1:3, 1:3] = np.random.rand(2, 2)
    outer_zero[1:3, 1:3] /= outer_zero[1:3, 1:3].sum()
    np.savetxt(f"imagesA/imageA_{i}.txt", outer_zero, fmt="%.4f")

    # Inner zero: inner pixels are zero, border pixels are random and normalized to 1
    inner_zero = np.zeros((4, 4), dtype=np.float32)
    inner_zero[0, :] = np.random.rand(4)
    inner_zero[3, :] = np.random.rand(4)
    inner_zero[:, 0] = np.random.rand(4)
    inner_zero[:, 3] = np.random.rand(4)
    inner_zero[1:3, 1:3] = 0
    inner_zero /= inner_zero.sum()
    np.savetxt(f"imagesB/imageB_{i}.txt", inner_zero, fmt="%.4f")