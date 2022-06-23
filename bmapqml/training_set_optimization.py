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

# Contains procedures for dealing with problematic (i.e. non-invertible or close to non-invertible)
# kernel matrices.


import numpy as np
from numba import njit, prange
from numba.types import bool_

class KernelUnstable(Exception):
    pass


@njit(fastmath=True)
def all_indices_except(to_include):
    num_left=0
    for el in to_include:
        if not el:
            num_left+=1
    output=np.zeros((num_left,), dtype=np.int32)
    arr_pos=0
    for el_id, el in enumerate(to_include):
        if not el:
#            print("Skipped: ", el_id)
            output[arr_pos]=el_id
            arr_pos+=1
    return output[:arr_pos]

@njit(fastmath=True, parallel=True)
def numba_linear_dependent_entries(train_kernel, residue_tol_coeff):
    num_elements=train_kernel.shape[0]

    sqnorm_residue=np.zeros(num_elements)
    residue_tolerance=np.zeros(num_elements)

    for i in prange(num_elements):
        sqnorm=train_kernel[i, i]
        sqnorm_residue[i]=sqnorm
        residue_tolerance[i]=sqnorm*residue_tol_coeff

    cur_orth_id=0

    to_include=np.ones(num_elements, dtype=bool_)

    orthonormalized_vectors=np.eye(num_elements)

    for cur_orth_id in range(num_elements):
        if not to_include[cur_orth_id]:
            continue
        # Normalize the vector.
        cur_norm=np.sqrt(sqnorm_residue[cur_orth_id])
        for i in prange(cur_orth_id+1):
            orthonormalized_vectors[cur_orth_id, i]/=cur_norm
        # Subtract projections of the normalized vector from all currently not orthonormalized vectors.
        # Also check that their residue is above the corresponding threshold.
        for i in prange(cur_orth_id+1, num_elements):
            if not to_include[i]:
                continue
            cur_product=0.0
            for j in range(cur_orth_id+1):
                if to_include[j]:
                    cur_product+=train_kernel[i, j]*orthonormalized_vectors[cur_orth_id, j]
            sqnorm_residue[i]-=cur_product**2
            if sqnorm_residue[i]<residue_tolerance[i]:
                to_include[i]=False
            else:
                for j in range(cur_orth_id+1):
                    orthonormalized_vectors[i, j]-=cur_product*orthonormalized_vectors[cur_orth_id, j]
        cur_orth_id+=1
    return all_indices_except(to_include)

# For distance-based cutting off redundant entries from the kernel matrix.
@njit(fastmath=True, parallel=True)
def kernel2sqdist(train_kernel):
    num_train=train_kernel.shape[0]
    sqdist_mat=np.zeros((num_train, num_train))
    for i in prange(num_train):
        for j in range(num_train):
            sqdist_mat[i,j]=train_kernel[i,i]+train_kernel[j,j]-2*train_kernel[i,j]
    return sqdist_mat

@njit(fastmath=True)
def min_id_sqdist(sqdist_row, to_include, entry_id):
    cur_min_sqdist=0.0
    cur_min_sqdist_id=0
    minimal_sqdist_init=False
    num_train=sqdist_row.shape[0]

    for j in range(num_train):
        if entry_id != j:
            cur_sqdist=sqdist_row[j]
            if (((cur_sqdist<cur_min_sqdist) or (not minimal_sqdist_init)) and to_include[j]):
                minimal_sqdist_init=True
                cur_min_sqdist=cur_sqdist
                cur_min_sqdist_id=j
    return cur_min_sqdist_id, cur_min_sqdist


@njit(fastmath=True, parallel=True)
def numba_rep_sqdist_mat(rep_arr):
    num_vecs=rep_arr.shape[0]
    sqdist_mat=np.zeros((num_vecs, num_vecs))
    for i in prange(num_vecs):
        for j in range(i):
            sqdist_mat[i, j]=np.sum(np.square(rep_arr[i]-rep_arr[j]))
            sqdist_mat[j, i]=sqdist_mat[i, j]
    return sqdist_mat

@njit(fastmath=True, parallel=True)
def numba_sqdist_exclude_nearest(sqdist_mat, min_sqdist, num_cut_closest_entries):
    num_train=sqdist_mat.shape[0]

    minimal_distance_ids=np.zeros(num_train, dtype=np.int32)
    minimal_distances=np.zeros(num_train)
    to_include=np.ones(num_train, dtype=bool_)

    for i in prange(num_train):
        minimal_distance_ids[i], minimal_distances[i]=min_id_sqdist(sqdist_mat[i], to_include, i)

    num_ignored=0

    while True:
        cur_min_id, cur_min_sqdist=min_id_sqdist(minimal_distances, to_include, -1)
        if (cur_min_sqdist > min_sqdist) and (min_sqdist > 0.0):
            break
        if np.random.random()>0.5:
            new_ignored=cur_min_id
        else:
            new_ignored=minimal_distance_ids[cur_min_id]

        to_include[new_ignored]=False
        num_ignored+=1
        if num_ignored==1:
            print("Smallest ignored distance:", cur_min_sqdist)
        if num_ignored==num_cut_closest_entries:
            print("Largest ignored distance:", cur_min_sqdist)
            break
        for i in prange(num_train):
            if to_include[i]:
                if (minimal_distance_ids[i]==new_ignored):
                    minimal_distance_ids[i], minimal_distances[i]=min_id_sqdist(sqdist_mat[i], to_include, i)

    return all_indices_except(to_include)



@njit(fastmath=True)
def numba_kernel_exclude_nearest(train_kernel, min_sqdist, num_cut_closest_entries):
    sqdist_mat=kernel2sqdist(train_kernel)
    return numba_sqdist_exclude_nearest(sqdist_mat, min_sqdist, num_cut_closest_entries)


