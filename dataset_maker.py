import numpy as np
import os

def generate_matrices(n):
    # Create output directories
    os.makedirs("./dataset/imagesA", exist_ok=True)
    os.makedirs("./dataset/imagesB", exist_ok=True)

    # Generate and save 100 "outer zero" and 100 "inner zero" matrices
    for i in range(100):
        # Outer zero: border is zero, inner pixels normalized to 1
        outer_zero = np.zeros((n, n), dtype=np.float32)
        if n > 2:
            inner_slice = slice(1, n-1)
            outer_zero[inner_slice, inner_slice] = np.random.rand(n-2, n-2)
            outer_zero[inner_slice, inner_slice] /= outer_zero[inner_slice, inner_slice].sum()
        np.savetxt(f"./dataset/imagesA/imageA_{i}.txt", outer_zero, fmt="%.4f")

        # Inner zero: inner pixels are zero, border pixels are random and normalized to 1
        inner_zero = np.zeros((n, n), dtype=np.float32)
        if n > 2:
            inner_zero[0, :] = np.random.rand(n)
            inner_zero[n-1, :] = np.random.rand(n)
            inner_zero[:, 0] = np.random.rand(n)
            inner_zero[:, n-1] = np.random.rand(n)
            inner_zero[1:n-1, 1:n-1] = 0
            inner_zero /= inner_zero.sum()
        np.savetxt(f"./dataset/imagesB/imageB_{i}.txt", inner_zero, fmt="%.4f")

# Example usage
generate_matrices(8)