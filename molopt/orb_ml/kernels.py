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


import numpy as np
import itertools, copy
from numba import njit, prange, float64, types, typed, typeof, deferred_type, int64, float32
from numba.experimental import jitclass
#   Note: using joblib here proved impossible due to jitclass objects not being pickable.
#from ..python_parallelization import embarassingly_parallel
from .oml_compound import OML_compound, OML_Slater_pair
import numpy as np

default_float=float64

def jitclass_list_type(item_constructor, *def_args):
    list_instance = typed.List()
    list_instance.append(item_constructor(*def_args))
    return typeof(list_instance)
    

def jitclass_list_wrho_specs_from_type(item_list_type):
    return [("num_components", int64), ("rhos", default_float[:]), ("components", item_list_type), ("norm_data", default_float[:])]

# Functions for linear kernels and normalization.

@njit(fastmath=True)
def arep_dot_product(inout_arr, scalar_rep1, scalar_rep2):
    inout_arr[0]=np.sum(np.exp(-(scalar_rep1-scalar_rep2)**2))


@njit(fastmath=True)
def arep_dot_product_wders(inout_arr, scalar_rep1, scalar_rep2):
    sqdiffs_arr=(scalar_rep1-scalar_rep2)**2
    inout_arr[0]=np.exp(-np.sum(sqdiffs_arr))
    inout_arr[1:]=inout_arr[0]*sqdiffs_arr

@njit(fastmath=True)
def weighted_dot_product(inout_arr, obj1, obj2, dot_prod_func, *args):
    inout_arr[:]=0.0
    temp_arr=np.copy(inout_arr)
    for i in range(obj1.num_components):
        for j in range(obj2.num_components):
            dot_prod_func(temp_arr, obj1.components[i], obj2.components[j], *args)
            inout_arr[:]+=obj1.rhos[i]*obj2.rhos[j]*temp_arr

@njit(fastmath=True)
def orb_orb_dot_product(inout_arr, orb1, orb2):
    weighted_dot_product(inout_arr, orb1, orb2, arep_dot_product)

@njit(fastmath=True)
def orb_orb_dot_product_wders(inout_arr, orb1, orb2):
    weighted_dot_product(inout_arr, orb1, orb2, arep_dot_product_wders)

@njit(fastmath=True)
def orb_orb_dot_product_norm(inout_arr, orb1, orb2):
    orb_orb_dot_product(inout_arr, orb1, orb2)
    inout_arr[0]*=orb1.norm_data[0]*orb2.norm_data[0]

@njit(fastmath=True)
def add_norm_der(inout_arr, obj1, obj2):
    inout_arr[1:]-=inout_arr[0]*(obj1.norm_data[1:]+obj2.norm_data[1:])
    inout_arr[:]*=obj1.norm_data[0]*obj2.norm_data[0]

@njit(fastmath=True)
def orb_orb_dot_product_norm_wders(inout_arr, orb1, orb2):
    weighted_dot_product(inout_arr, orb1, orb2, arep_dot_product_wders)
    add_norm_der(inout_arr, orb1, orb2)

@njit(fastmath=True)
def mol_mol_dot_product(inout_arr, mol1, mol2):
    weighted_dot_product(inout_arr, mol1, mol2, orb_orb_dot_product_norm)

@njit(fastmath=True)
def mol_mol_dot_product_wders(inout_arr, mol1, mol2):
    weighted_dot_product(inout_arr, mol1, mol2, orb_orb_dot_product_norm_wders)

@njit(fastmath=True)
def mol_mol_dot_product_norm(inout_arr, mol1, mol2):
    mol_mol_dot_product(inout_arr, mol1, mol2)
    inout_arr[:]*=mol1.norm_data[0]*mol2.norm_data[0]

@njit(fastmath=True)
def mol_mol_dot_product_norm_wders(inout_arr, mol1, mol2):
    mol_mol_dot_product_wders(inout_arr, mol1, mol2)
    add_norm_der(inout_arr, mol1, mol2)

@njit(fastmath=True)
def init_obj_norm(obj, dot_prod_subroutine, with_ders):
    dot_prod_subroutine(obj.norm_data, obj, obj)
    if with_ders:
        obj.norm_data[1:]/=obj.norm_data[0]*2
    obj.norm_data[0]=np.sqrt(obj.norm_data[0])**(-1)

# Classes used in numba implementation.
@jitclass(jitclass_list_wrho_specs_from_type(default_float[:,:]))
class Orb_rep_numba:
    def __init__(self, num_components, num_scalar_reps, num_norm_comps):
        self.components=np.empty((num_components, num_scalar_reps), dtype=default_float)
        self.num_components=num_components
        self.rhos=np.empty((num_components,), dtype=default_float)
        self.norm_data=np.empty((num_norm_comps,), dtype=default_float)

Orb_rep_numba_list_type=jitclass_list_type(Orb_rep_numba, 1, 1, 1)

@jitclass(jitclass_list_wrho_specs_from_type(Orb_rep_numba_list_type))
class Mol_rep_numba:
    def __init__(self, components, num_components, num_norm_comps):
        self.components=components
        self.num_components=num_components
        self.rhos=np.empty((num_components,), dtype=default_float)
        self.norm_data=np.empty((num_norm_comps,), dtype=default_float)

# Procedure for initializing normalization data in Mol_rep_numba list.
@njit(fastmath=True, parallel=True)
def numba_list_init_norm(mol_list, list_length, orb_orb_dp, mol_mol_dp, with_ders, global_Gauss):
    for i in prange(list_length):
        mol=mol_list[i]
        for j in range(mol.num_components):
            init_obj_norm(mol.components[j], orb_orb_dp, with_ders)
        if global_Gauss:
            init_obj_norm(mol, mol_mol_dp, with_ders)

def list_init_norm(mol_list, with_ders=False, global_Gauss=False):
    if with_ders:
        mol_mol_dp=mol_mol_dot_product_wders
        orb_orb_dp=orb_orb_dot_product_wders
    else:
        mol_mol_dp=mol_mol_dot_product
        orb_orb_dp=orb_orb_dot_product
    numba_list_init_norm(mol_list, len(mol_list), orb_orb_dp, mol_mol_dp, with_ders, global_Gauss)

# Constructors from objects used in the rest of the code.
def gen_input_Orb_rep_numba(orb_rep, resc_params, with_ders=False):
    num_scalar_reps=orb_rep.orb_atom_reps[0].scalar_reps.shape[0]
    num_norm_comps=1
    if with_ders:
        num_norm_comps+=num_scalar_reps
    num_orb_atom_reps=len(orb_rep.orb_atom_reps)
    output=Orb_rep_numba(num_orb_atom_reps, num_scalar_reps, num_norm_comps)
    for orb_id, orb_atom_rep in enumerate(orb_rep.orb_atom_reps):
        output.components[orb_id, :]=orb_atom_rep.scalar_reps[:]
        output.rhos[orb_id]=orb_atom_rep.rho
        if resc_params is not None:
            output.components[orb_id, :]*=resc_params
    return output

def is_pair_rep(comp):
    return (hasattr(comp, 'comps'))

def iterated_orb_reps(oml_comp, pair_rep=None, single_orb_list=False):
    if pair_rep is None:
        pair_rep=is_pair_rep(oml_comp)
    if pair_rep:
        return list(itertools.chain(oml_comp.comps[0].orb_reps, oml_comp.comps[1].orb_reps))
    else:
        if single_orb_list:
            return [oml_comp]
        else:
            return oml_comp.orb_reps

def gen_input_Mol_rep_numba(oml_comp, resc_params, with_ders=False, global_Gauss=False):
    pair_rep=is_pair_rep(oml_comp)
    orb_reps=iterated_orb_reps(oml_comp, pair_rep=pair_rep)
    if global_Gauss:
        num_norm_comps=1
        if with_ders:
            num_norm_comps+=orb_reps[0].orb_atom_reps[0].scalar_reps.shape[0]
    else:
        num_norm_comps=0
    num_components=len(orb_reps)
    components=typed.List()
    rhos=np.zeros(num_components)
    for orb_id, orb_rep in enumerate(orb_reps):
        components.append(gen_input_Orb_rep_numba(orb_rep, resc_params, with_ders=with_ders))
        rhos[orb_id]=orb_rep.rho
    if pair_rep:
        rhos[:len(oml_comp.comps[0].orb_reps)]*=-1
    output=Mol_rep_numba(components, num_components, num_norm_comps)
    output.rhos[:]=rhos[:]
    return output

# Creating input for the kernel function.

def gen_sep_orb_kern_input(oml_compound_array, sigmas, with_ders=False, global_Gauss=False):
    if sigmas is None:
        resc_params=None
    else:
        resc_params=.5/sigmas
    converted_comps=typed.List([gen_input_Mol_rep_numba(oml_comp, resc_params, with_ders=with_ders, global_Gauss=global_Gauss) for oml_comp in oml_compound_array])
#    converted_comps=[gen_input_Mol_rep_numba(oml_comp, resc_params, with_ders=with_ders, global_Gauss=global_Gauss) for oml_comp in oml_compound_array]
    list_init_norm(converted_comps, with_ders=with_ders, global_Gauss=global_Gauss)
    return converted_comps


# Functions for Gaussian kernel function calculation.
@njit(fastmath=True)
def lin2gauss(lin_cov, inv_sq_global_sigma):
    return np.exp(-(1.-lin_cov)*inv_sq_global_sigma)

@njit(fastmath=True)
def lin2gauss_wders(lin_cov_wders, inv_sq_global_sigma):
    lin_cov_wders[0]=lin2gauss(lin_cov_wders[1], inv_sq_global_sigma)
    lin_cov_wders[1]=lin_cov_wders[0]*(1.-lin_cov_wders[1])
    lin_cov_wders[2:]*=lin_cov_wders[0]*inv_sq_global_sigma

@njit(fastmath=True)
def orb_orb_Gauss(inout_arr, orb1, orb2, inv_sq_sigma):
    orb_orb_cov_norm(inout_arr, orb1, orb2)
    inout_arr[0]=lin2gauss(inout_arr[0], inv_sq_sigma)

@njit(fastmath=True)
def orb_orb_Gauss_wders(inout_arr, orb1, orb2, inv_sq_sigma):
    orb_orb_cov_norm_wders(inout_arr[1:], orb1, orb2)
    lin2gauss_wders(inout_arr, inv_sq_sigma)

@njit(fastmath=True)
def mol_mol_Gauss_dot_product(inout_arr, mol1, mol2, inv_sq_sigma):
    weighted_dot_product(inout_arr, mol1, mol2, orb_orb_Gauss, inv_sq_sigma)

@njit(fastmath=True)
def mol_mol_Gauss_dot_product_wders(inout_arr, mol1, mol2, inv_sq_sigma):
    weighted_dot_product(inout_arr, mol1, mol2, orb_orb_Gauss_wders, inv_sq_sigma)

@njit(fastmath=True)
def mol_mol_global_Gauss(inout_arr, mol1, mol2, inv_sq_sigma):
    mol_mol_dot_product_norm(inout_arr, mol1, mol2)
    inout_arr[0]=lin2gauss(inout_arr[0], inv_sq_sigma)

@njit(fastmath=True)
def mol_mol_global_Gauss_wders(inout_arr, mol1, mol2, inv_sq_sigma):
    mol_mol_dot_product_norm_wders(inout_arr[1:], mol1, mol2)
    lin2gauss_wders(inout_arr, inv_sq_sigma)

# Kernel function calculation.
@njit(fastmath=True, parallel=True)
def numba_kernel_matrix(inout_arr, mol_arr1, mol_arr2, num_mols1, num_mols2, inv_sq_sigma, kernel_el_func, sym_kernel, *extra_args):
    for i1 in prange(num_mols1):
        if sym_kernel:
            upper_mol2=i1+1
        else:
            upper_mol2=num_mols2
        for i2 in range(upper_mol2):
            kernel_el_func(inout_arr[i1, i2:i2+1], mol_arr1[i1], mol_arr2[i2], *extra_args)
    if sym_kernel:
        for i1 in prange(num_mols1):
            for i2 in range(i1+1):
                inout_arr[i2, i1]=inout_arr[i1, i2]

@njit(fastmath=True, parallel=True)
def numba_kernel_matrix_wders(inout_arr, mol_arr1, mol_arr2, num_mols1, num_mols2, kernel_el_func, der_resc_arr, sym_kernel, *extra_args):
    for i1 in prange(num_mols1):
        if sym_kernel:
            upper_mol2=i1+1
        else:
            upper_mol2=num_mols2
        for i2 in range(upper_mol2):
            kernel_el_func(inout_arr[i1, i2], mol_arr1[i1], mol_arr2[i2], *extra_args)
            inout_arr[i1, i2, 1:]*=der_resc_arr
    if sym_kernel:
        for i1 in prange(num_mols1):
            for i2 in range(i1+1):
                inout_arr[i2, i1, :]=inout_arr[i1, i2, :]

# Interfacing with the rest of the code.
def kernel_matrix(comp_arr1, comp_arr2, sigmas, kernel_el_func, with_ders=False, linear_kernel=True, global_Gauss=False):
    sym_kernel=(comp_arr2 is None)
    num_mols1=len(comp_arr1)

    if linear_kernel:
        lin_sigmas=sigmas
        extra_args=()
    else:
        lin_sigmas=sigmas[1:]
        extra_args=(sigmas[0]**[-2])

    if with_ders:
        der_resc=2.0/lin_sigmas
        if not linear_kernel:
            der_resc=np.array([2.0/sigmas[0]**3, *der_resc])

    numba_comp_arr1=gen_sep_orb_kern_input(comp_arr1, lin_sigmas, with_ders=with_ders, global_Gauss=global_Gauss)
    if sym_kernel:
        numba_comp_arr2=numba_comp_arr1
        num_mols2=num_mols1
    else:
        numba_comp_arr2=gen_sep_orb_kern_input(comp_arr2, lin_sigmas, with_ders=with_ders, global_Gauss=global_Gauss)
        num_mols2=len(comp_arr2)
    if with_ders:
        output=np.empty((num_mols1, num_mols2, 1+len(sigmas)))
        der_resc_arr=np.array([2.0/sigmas[0]**3, *2.0/sigmas[1:]])
        numba_kernel_matrix_wders(output, numba_comp_arr1, numba_comp_arr2, num_mols1, num_mols2, kernel_el_func, der_resc_arr, sym_kernel, *extra_args)
    else:
        output=np.empty((num_mols1, num_mols2), dtype=default_float)
        numba_kernel_matrix(output, numba_comp_arr1, numba_comp_arr2, num_mols1, num_mols2, kernel_el_func, sym_kernel, *extra_args)
    return output

# Auxiliary for hyperparameter optimization.
@njit(fastmath=True)
def find_orb_vec_rep_moments(comp_list, moment_list):
    num_mols=len(comp_list)
    num_moments=moment_list.shape[0]
    num_scalar_reps=comp_list[0].components[0].components.shape[-1]

    output=np.zeros((num_moments, num_scalar_reps))

    norm_const=0.0
    for mol_id in prange(num_mols):
        mol=comp_list[mol_id]
        for orb_id in range(mol.num_components):
            orb=mol.components[orb_id]
            for arep_id in range(orb.num_components):
                cur_rho=np.abs(mol.rhos[orb_id]*orb.rhos[arep_id])
                norm_const+=cur_rho
                for moment_id in range(num_moments):
                    output[moment_id, :]+=cur_rho*orb.components[arep_id]**moment_list[moment_id]
    return output/norm_const

def oml_ensemble_avs_stddevs(compound_list):
    if (isinstance(compound_list[0], OML_compound) or isinstance(compound_list[0], OML_Slater_pair)):
        compound_list_converted=gen_sep_orb_kern_input(compound_list, None)
    else:
        compound_list_converted=compound_list

    moment_vals=find_orb_vec_rep_moments(compound_list_converted, np.array([1, 2]))
    avs=moment_vals[0]
    avs2=moment_vals[1]
    stddevs=np.sqrt(avs2-avs**2)
    return avs, stddevs


### For linear kernel with separable orbs.

def lin_sep_orb_kernel(A, B, sigmas, with_ders=False):
    if with_ders:
        kernel_element_func=mol_mol_dot_product_wders
    else:
        kernel_element_func=mol_mol_dot_product
    return kernel_matrix(A, B, sigmas, kernel_element_func, with_ders=with_ders)

def lin_sep_orb_sym_kernel(A, sigmas, **kwargs):
    return lin_sep_orb_kernel(A, None, sigmas, **kwargs)

def gauss_sep_orb_kernel(A, B, sigmas, with_ders=False, global_Gauss=False):
    if with_ders:
        if global_Gauss:
            kernel_element_func=mol_mol_global_Gauss_wders
        else:
            kernel_element_func=mol_mol_Gauss_dot_product_wders
    else:
        if global_Gauss:
            kernel_element_func=mol_mol_global_Gauss
        else:
            kernel_element_func=mol_mol_Gauss_dot_product
    return kernel_matrix(A, B, sigmas, kernel_element_func, with_ders=with_ders, linear_kernel=False, global_Gauss=global_Gauss)

def gauss_sep_orb_sym_kernel(A, sigmas, **kwargs):
    return gauss_sep_orb_kernel(A, None, sigmas, **kwargs)

