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


# TO-DO perhaps in the future add options for reduced hyperparameter derivatives calculated directly here
# rather than converted to.


import numpy as np
from numba import njit, prange

@njit(fastmath=True)
def l1_norm(vec1, vec2, inv_sigmas):
    return np.sum(np.abs(vec1-vec2)*inv_sigmas)

@njit(fastmath=True)
def l2_half_sq_norm(vec1, vec2, inv_sq_sigmas):
    return np.sum(inv_sq_sigmas*(vec1-vec2)**2)*0.5

@njit(fastmath=True)
def kernel_element(vec1, vec2, dist_func, dist_params):
    return np.exp(-dist_func(vec1, vec2, dist_params))

@njit(fastmath=True)
def kernel_element_wders(vec1, vec2, dist_func, dist_params, inout_arr):
    exp_arg=dist_func(vec1, vec2, dist_params)
    inout_arr[0]=np.exp(-exp_arg)
    inout_arr[1]=-exp_arg*inout_arr[0]

@njit(fastmath=True, parallel=True)
def kernel_matrix_noders(A, B, dist_func, dist_params):
    output_kernel=np.zeros((A.shape[0], B.shape[0]))
    for i in prange(A.shape[0]):
        for j in range(B.shape[0]):
            output_kernel[i, j]=kernel_element(A[i], B[j], dist_func, dist_params)
    return output_kernel

@njit(fastmath=True, parallel=True)
def kernel_matrix_wders(A, B, dist_func, dist_params):
    output_kernel=np.zeros((A.shape[0], B.shape[0], 2))
    for i in prange(A.shape[0]):
        for j in range(B.shape[0]):
            kernel_element_wders(A[i], B[j], dist_func, dist_params, output_kernel[i, j, :])


@njit(fastmath=True, parallel=True)
def sym_kernel_matrix_noders(A, dist_func, dist_params):
    output_kernel=np.zeros((A.shape[0], A.shape[0]))
    for i in prange(A.shape[0]):
        for j in range(i+1):
            output_kernel[i, j]=kernel_element(A[i], A[j], dist_func, dist_params)
            output_kernel[j, i]=output_kernel[i, j]
    return output_kernel

#@njit(fastmath=True, parallel=True)
def sym_kernel_matrix_wders(A, dist_func, dist_params):
    output_kernel=np.zeros((A.shape[0], A.shape[0], 2))
    for i in prange(A.shape[0]):
        for j in range(i+1):
            kernel_element_wders(A[i], A[j], dist_func, dist_params, output_kernel[i, j, :])
            output_kernel[j, i, :]=output_kernel[i, j, :]
    return output_kernel

def kernel_matrix(A, B, dist_func, dist_params, with_ders=False):
    if B is None:
        if with_ders:
            return sym_kernel_matrix_wders(A, dist_func, dist_params)
        else:
            return sym_kernel_matrix_noders(A, dist_func, dist_params)
    else:
        if with_ders:
            return kernel_matrix_wders(A, B, dist_func, dist_params)
        else:
            return kernel_matrix_noders(A, B, dist_func, dist_params)

def gaussian_kernel_matrix(A, B, sigma_arr, with_ders=False, sym_kernel=False):
    inv_sq_sigmas=np.zeros((A.shape[-1],))
    inv_sq_sigmas[:]=sigma_arr[0]**(-2)
    dist_func=l2_half_sq_norm
    output_kernel=kernel_matrix(A, B, dist_func, inv_sq_sigmas, with_ders=with_ders)
    if with_ders:
        output_kernel[:, :, 1]*=-2.0/sigma_arr[0]
    return output_kernel

def laplacian_kernel_matrix(A, B, sigma_arr, with_ders=False, sym_kernel=False):
    inv_sigmas=np.zeros((A.shape[-1],))
    inv_sigmas[:]=sigma_arr[0]**(-1)
    dist_func=l1_norm
    output_kernel=kernel_matrix(A, B, dist_func, inv_sigmas, with_ders=with_ders)
    if with_ders:
        output_kernel[:, :, 1]*=sigma_arr[0]**(-1)
    return output_kernel

def gaussian_sym_kernel_matrix(A, sigma_arr, with_ders=False):
    return gaussian_kernel_matrix(A, None, sigma_arr, with_ders=with_ders)

def laplacian_sym_kernel_matrix(A, sigma_arr, with_ders=False):
    return laplacian_kernel_matrix(A, None, sigma_arr, with_ders=with_ders)


class symmetrized_kernel_matrix:
    def __init__(self, kernel_func):
        self.kernel_func=kernel_func
    def __call__(self, A, *args, **kwargs):
        return self.kernel_func(A, A, *args, **kwargs)

