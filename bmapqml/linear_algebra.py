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

# For dealing with several Cholesky decompositions at once.

import numpy as np
from .utils import where2slice, nullify_ignored
from scipy.linalg import cho_factor, cho_solve

def scipy_cho_solve(mat, vec):
    c, low=cho_factor(mat)
    return cho_solve((c, low), vec)


def nullify_ignored(arr, indices_to_ignore):
    if indices_to_ignore is not None:
        for row_id, cur_ignore_indices in enumerate(indices_to_ignore):
            arr[row_id][where2slice(np.logical_not(cur_ignore_indices))]=0.0

class Cho_multi_factors:
    def __init__(self, train_kernel, indices_to_ignore=None, ignored_orderable=False):
        self.indices_to_ignore=indices_to_ignore
        self.ignored_orderable=ignored_orderable
        self.single_cho_decomp=(indices_to_ignore is None)
        if not self.single_cho_decomp:
            self.single_cho_decomp=(not self.indices_to_ignore.any())
        if self.single_cho_decomp:
            self.cho_factors=[cho_factor(train_kernel)]
        else:
            if self.ignored_orderable:
                ignored_nums=[]
                for i, cur_ignored in enumerate(self.indices_to_ignore):
                    ignored_nums.append((i, np.sum(cur_ignored)))
                ignored_nums.sort(key = lambda x : x[1])
                self.availability_order=np.array([i[0] for i in ignored_nums])
                self.avail_quant_nums=[self.indices_to_ignore.shape[0]-np.sum(cur_ignored) for cur_ignored in self.indices_to_ignore.T]
                self.cho_factors=[cho_factor(train_kernel[self.availability_order, :][:, self.availability_order])]
            else:
                self.cho_factors=[]
                for cur_ignore_ids in self.indices_to_ignore.T:
                    s=where2slice(cur_ignore_ids)
                    self.cho_factors.append(cho_factor(train_kernel[s, :][:, s]))
    def solve_with(self, rhs):
        if len(rhs.shape)==1:
            assert(self.single_cho_decomp)
            cycled_rhs=np.array([rhs])
        else:
            if not self.single_cho_decomp:
                if self.ignored_orderable:
                    assert(len(self.avail_quant_nums)==rhs.shape[1])
                else:
                    assert(len(self.cho_factors)==rhs.shape[1])
            cycled_rhs=rhs.T
        output=np.zeros(cycled_rhs.shape)
        for rhs_id, rhs_component in enumerate(cycled_rhs):
            if self.indices_to_ignore is None:
                included_indices=np.array(range(len(rhs_component)))
            else:
                if self.ignored_orderable:
                    included_indices=self.availability_order[:self.avail_quant_nums[rhs_id]]
                else:
                    included_indices=where2slice(self.indices_to_ignore[:, rhs_id])
            if self.single_cho_decomp:
                cur_decomp=self.cho_factors[0]
            else:
                if self.ignored_orderable:
                    cur_decomp=(self.cho_factors[0][0][:self.avail_quant_nums[rhs_id], :][:, :self.avail_quant_nums[rhs_id]], self.cho_factors[0][1])
                else:
                    cur_decomp=self.cho_factors[rhs_id]
            output[rhs_id, included_indices]=cho_solve(cur_decomp, rhs_component[included_indices])
        if len(rhs.shape)==1:
            return output[0]
        else:
            return output

