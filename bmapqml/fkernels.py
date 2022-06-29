# MIT License
#
# Copyright (c) 2021-2022 Konstantin Karandashev
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# TO-DO: if this becomes the dominant way for doing things 

import numpy as np
from .precompilation import precompiled
precompiled("ffkernels")
from .ffkernels import fgaussian_kernel_matrix, fgaussian_sym_kernel_matrix
from .ffkernels import flaplacian_kernel
from .ffkernels import flaplacian_kernel_symmetric

def gaussian_kernel_matrix(A, B, sigmas, with_ders=False):
    if with_ders:
        num_kern_comps=2
    else:
        num_kern_comps=1

    num_features=A.shape[1]

    num_A=A.shape[0]
    num_B=B.shape[0]

    assert(num_features==B.shape[1])

    kernel_matrix=np.zeros((num_A, num_B, num_kern_comps), dtype=float)

    fgaussian_kernel_matrix(A.T, B.T, sigmas[0], num_kern_comps, num_A, num_B, num_features, kernel_matrix.T)

    if with_ders:
        return kernel_matrix
    else:
        return kernel_matrix[:, :, 0]

def gaussian_sym_kernel_matrix(A, sigmas, with_ders=False):

    if with_ders:
        num_kern_comps=2
    else:
        num_kern_comps=1

    num_features=A.shape[1]

    num_A=A.shape[0]

    kernel_matrix=np.zeros((num_A, num_A, num_kern_comps), dtype=float)

    fgaussian_sym_kernel_matrix(A.T, sigmas[0], num_kern_comps, num_A, num_features, kernel_matrix.T)

    if with_ders:
        return kernel_matrix
    else:
        return kernel_matrix[:, :, 0]






def laplacian_kernel(A, B, sigma):
    """ Calculates the Laplacian kernel matrix K, where :math:`K_{ij}`:
            :math:`K_{ij} = \\exp \\big( -\\frac{\\|A_i - B_j\\|_1}{\sigma} \\big)`
        Where :math:`A_{i}` and :math:`B_{j}` are representation vectors.
        K is calculated using an OpenMP parallel Fortran routine.
        :param A: 2D array of representations - shape (N, representation size).
        :type A: numpy array
        :param B: 2D array of representations - shape (M, representation size).
        :type B: numpy array
        :param sigma: The value of sigma in the kernel matrix.
        :type sigma: float
        :return: The Laplacian kernel matrix - shape (N, M)
        :rtype: numpy array
    """

    na = A.shape[0]
    nb = B.shape[0]

    K = np.empty((na, nb), order='F')

    # Note: Transposed for Fortran
    flaplacian_kernel(A.T, na, B.T, nb, K, sigma)

    return K

def laplacian_kernel_symmetric(A, sigma):
    """ Calculates the symmetric Laplacian kernel matrix K, where :math:`K_{ij}`:
            :math:`K_{ij} = \\exp \\big( -\\frac{\\|A_i - A_j\\|_1}{\sigma} \\big)`
        Where :math:`A_{i}` are representation vectors.
        K is calculated using an OpenMP parallel Fortran routine.
        :param A: 2D array of representations - shape (N, representation size).
        :type A: numpy array
        :param sigma: The value of sigma in the kernel matrix.
        :type sigma: float
        :return: The Laplacian kernel matrix - shape (N, N)
        :rtype: numpy array
    """

    na = A.shape[0]

    K = np.empty((na, na), order='F')

    # Note: Transposed for Fortran
    flaplacian_kernel_symmetric(A.T, na, K, sigma)

    return K
