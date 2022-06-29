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



