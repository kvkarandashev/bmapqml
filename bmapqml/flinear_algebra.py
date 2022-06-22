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

from .precompilation import precompiled
precompiled("fflinear_algebra")
from .fflinear_algebra import flinear_dependent_entries
import numpy as np

def linear_dependent_entries(train_kernel, residue_tol_coeff=.0, use_Fortran=False, lambda_val=0.0,
                                    return_orthonormalized=False, ascending_residue_order=True):

    num_elements=train_kernel.shape[0]
    output_indices=np.zeros(num_elements, dtype=np.int32)
    orthonormalized_vectors=np.zeros((num_elements, num_elements))
    flinear_dependent_entries(train_kernel, orthonormalized_vectors.T, num_elements,
                    residue_tol_coeff, lambda_val, ascending_residue_order, output_indices)
    if output_indices[0]==-2:
        raise KernelUnstable
    final_output=[]
    for i in range(num_elements):
        if output_indices[i]==-1:
            final_output=output_indices[:i]
            break
    if return_orthonormalized:
        return final_output, orthonormalized_vectors
    else:
        return final_output


def solve_Gram_Schmidt(sym_mat, vec, residue_tol_coeff=.0, lambda_val=0.0, ascending_residue_order=False, use_Fortran=False):
    ignored_indices, orthonormalized_vectors=linear_dependent_entries(sym_mat, residue_tol_coeff, use_Fortran=use_Fortran,
                    lambda_val=lambda_val, return_orthonormalized=True, ascending_residue_order=ascending_residue_order)
    return ignored_indices, np.matmul(orthonormalized_vectors.T, np.matmul(orthonormalized_vectors, vec))

