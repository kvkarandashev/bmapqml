# Demonstrates how solve_Gram_Schmidt can be used to find alphas in cases
# where the kernel matrix is close to being singular despite addition of lambda.

import numpy as np
from numpy.linalg import LinAlgError
import random
from bmapqml.linear_algebra import scipy_cho_solve
from bmapqml.flinear_algebra import solve_Gram_Schmidt

np.random.seed(1)

def print_for(feature_num, vec_num):
    print("Number of vectors:", vec_num)
    print("Number of features:", feature_num)
    quantities=np.random.random((vec_num,))
    feature_vectors=np.random.random((vec_num, feature_num))
    kernel_matrix=np.matmul(feature_vectors, feature_vectors.T)
    GS_ignored_indices, alphas2=solve_Gram_Schmidt(kernel_matrix, quantities, residue_tol_coeff=1.e-9)
    print("Number of ignored indices:", len(GS_ignored_indices))

    # Trying to find alphas with scipy-based procedures.
    try:
        alphas1=scipy_cho_solve(kernel_matrix, quantities)
        print("Alphas difference between GS and SciPy:", np.sum(np.abs(alphas1-alphas2)))
    except LinAlgError:
        print("scipy_cho_solve failed")


print_for(20, 20)
print_for(20, 50)
