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
from ..precompilation import precompiled
precompiled("orb_ml.ffkernels")
from .ffkernels import fgmo_sep_orb_sym_kernel_wders, fgmo_sep_orb_kernel_wders
from .kernels import is_pair_rep, iterated_orb_reps

class fGMO_sep_orb_kern_input:
    def __init__(self, oml_compound_array=None, pair_rep=None):
        if pair_rep is None:
            pair_rep=is_pair_rep(oml_compound_array[0])
        if pair_rep:
            self.max_num_scalar_reps=len(oml_compound_array[0].comps[0].orb_reps[0].orb_atom_reps[0].scalar_reps)
        else:
            self.max_num_scalar_reps=len(oml_compound_array[0].orb_reps[0].orb_atom_reps[0].scalar_reps)

        self.num_mols=len(oml_compound_array)
        self.orb_nums=np.zeros((self.num_mols,), dtype=int)
        self.max_num_orbs=0

        for comp_id, oml_comp in enumerate(oml_compound_array):
            orb_list=iterated_orb_reps(oml_comp, pair_rep=pair_rep)
            self.orb_nums[comp_id]=len(orb_list)

        self.max_num_orbs=max(self.orb_nums)
        self.orb_atom_nums=np.zeros((self.num_mols, self.max_num_orbs), dtype=int)
        self.orb_rhos=np.zeros((self.num_mols, self.max_num_orbs))
        for comp_id, oml_comp in enumerate(oml_compound_array):
            orb_list=iterated_orb_reps(oml_comp, pair_rep=pair_rep)
            for orb_id, orb_rep in enumerate(orb_list):
                self.orb_atom_nums[comp_id, orb_id]=len(orb_rep.orb_atom_reps)
                self.orb_rhos[comp_id, orb_id]=orb_rep.rho
            if pair_rep:
                self.orb_rhos[comp_id, :len(oml_comp.comps[0].orb_reps)]*=-1

        self.max_num_orb_atom_reps=np.amax(self.orb_atom_nums)

        self.orb_arep_rhos=np.zeros((self.num_mols, self.max_num_orbs, self.max_num_orb_atom_reps))
        self.orb_atom_sreps=np.zeros((self.num_mols, self.max_num_orbs, self.max_num_orb_atom_reps, self.max_num_scalar_reps))
        for ind_comp, oml_comp in enumerate(oml_compound_array):
            for ind_orb, orb_rep in enumerate(iterated_orb_reps(oml_comp, pair_rep=pair_rep)):
                for ind_orb_arep, orb_arep in enumerate(orb_rep.orb_atom_reps):
                    self.orb_arep_rhos[ind_comp, ind_orb, ind_orb_arep]=orb_arep.rho
                    self.orb_atom_sreps[ind_comp, ind_orb, ind_orb_arep, :]=orb_arep.scalar_reps[:]



def gauss_sep_orb_kernel_conv(Ac, Bc, sigmas, preserve_converted_arrays=True, with_ders=False, global_Gauss=False):
    if with_ders:
        num_kern_comps=1+len(sigmas)
    else:
        num_kern_comps=1
    kernel_mat = np.zeros((Ac.num_mols, Bc.num_mols, num_kern_comps))
    fgmo_sep_orb_kernel_wders(Ac.max_num_scalar_reps,
                Ac.orb_atom_sreps.T, Ac.orb_arep_rhos.T, Ac.orb_rhos.T,
                Ac.orb_atom_nums.T, Ac.orb_nums,
                Ac.max_num_orb_atom_reps, Ac.max_num_orbs, Ac.num_mols,
                Bc.orb_atom_sreps.T, Bc.orb_arep_rhos.T, Bc.orb_rhos.T,
                Bc.orb_atom_nums.T, Bc.orb_nums,
                Bc.max_num_orb_atom_reps, Bc.max_num_orbs, Bc.num_mols,
                sigmas, global_Gauss, kernel_mat.T, num_kern_comps)
    if with_ders:
        return kernel_mat
    else:
        return kernel_mat[:, :, 0]


def gauss_sep_orb_kernel(A, B, sigmas, with_ders=False, global_Gauss=False):
    Ac=fGMO_sep_orb_kern_input(oml_compound_array=A)
    Bc=fGMO_sep_orb_kern_input(oml_compound_array=B)
    return gauss_sep_orb_kernel_conv(Ac, Bc, sigmas, with_ders=with_ders, global_Gauss=global_Gauss)

def gauss_sep_orb_sym_kernel_conv(Ac, sigmas, with_ders=False, global_Gauss=False):
    if with_ders:
        num_kern_comps=1+len(sigmas)
    else:
        num_kern_comps=1

    assert(Ac.max_num_scalar_reps+1==len(sigmas))

    kernel_mat = np.zeros((Ac.num_mols, Ac.num_mols, num_kern_comps))
    fgmo_sep_orb_sym_kernel_wders(Ac.max_num_scalar_reps,
                Ac.orb_atom_sreps.T, Ac.orb_arep_rhos.T, Ac.orb_rhos.T,
                Ac.orb_atom_nums.T, Ac.orb_nums,
                Ac.max_num_orb_atom_reps, Ac.max_num_orbs, Ac.num_mols,
                sigmas, global_Gauss, kernel_mat.T, num_kern_comps)

    if with_ders:
        return kernel_mat
    else:
        return kernel_mat[:, :, 0]

def gauss_sep_orb_sym_kernel(A, sigmas, with_ders=False, global_Gauss=False):
    Ac=fGMO_sep_orb_kern_input(oml_compound_array=A)
    return gauss_sep_orb_sym_kernel_conv(Ac, sigmas, with_ders=with_ders, global_Gauss=global_Gauss)

