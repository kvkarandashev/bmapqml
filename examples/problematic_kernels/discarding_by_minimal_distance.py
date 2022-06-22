import numpy as np
from bmapqml.training_set_optimization import numba_kernel_exclude_nearest

np.random.seed(1)

def find_minimal_sqdistance(vectors):
    min_sqdist=None
    for i1, vec1 in enumerate(vectors):
        for vec2 in vectors[:i1]:
            cur_sqdist=np.sum((vec1-vec2)**2)
            if (min_sqdist is None) or (cur_sqdist<min_sqdist):
                min_sqdist=cur_sqdist
    return min_sqdist

num_features=3
num_vectors=500

vectors=np.random.random((num_vectors, num_features))

kernel_matrix=np.matmul(vectors, vectors.T)

print("Original minimal distance:", find_minimal_sqdistance(vectors))

sparsed_indices1=numba_kernel_exclude_nearest(kernel_matrix, .0, 50)
sparsed_indices2=numba_kernel_exclude_nearest(kernel_matrix, 1.e-3, 0)

print("Sparced by deleting 50 closest neighbors:", find_minimal_sqdistance(vectors[sparsed_indices1]))
print("Sparced by deleting neighbors closer than 1.e-3", find_minimal_sqdistance(vectors[sparsed_indices2]))
