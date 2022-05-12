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


from numba import njit, prange
import numpy as np
from .aux_classes import generate_ao_arr

# Some auxiliary numba functions.
@njit(fastmath=True)
def gen_orb_atom_scalar_rep(coup_mats, coeffs, angular_momenta, atom_ao_range, max_ang_mom, scalar_reps):

    cur_array_position=0
    num_coup_mats=coup_mats.shape[0]
    scalar_reps[:]=0.0

    for mat_counter in range(num_coup_mats):
        for same_atom_check in range(2):
            for ang_mom1 in range(max_ang_mom+1):
                for ang_mom2 in range(max_ang_mom+1):
                    if not ((same_atom_check==0) and (ang_mom1 > ang_mom2)):
                        add_orb_atom_coupling(coup_mats[mat_counter], coeffs, angular_momenta, atom_ao_range,
                                same_atom_check, ang_mom1, ang_mom2, scalar_reps[cur_array_position:cur_array_position+1])
                        cur_array_position=cur_array_position+1

@njit(fastmath=True)
def add_orb_atom_coupling(mat, coeffs, angular_momenta, atom_ao_range, same_atom, ang_mom1, ang_mom2, cur_coupling_val):

    for ao1 in range(atom_ao_range[0], atom_ao_range[1]):
        if angular_momenta[ao1]==ang_mom1:
            if (same_atom==0):
                add_aos_coupling(mat, coeffs, angular_momenta, ang_mom2, atom_ao_range[0], atom_ao_range[1], ao1, cur_coupling_val)
            else:
                add_aos_coupling(mat, coeffs, angular_momenta, ang_mom2, 0, atom_ao_range[0], ao1, cur_coupling_val)
                add_aos_coupling(mat, coeffs, angular_momenta, ang_mom2, atom_ao_range[1], coeffs.shape[0], ao1, cur_coupling_val)

@njit(fastmath=True)
def add_aos_coupling(mat, coeffs, angular_momenta, ang_mom2, lower_index, upper_index, ao1, cur_coupling_val):

    for ao2 in range(lower_index, upper_index):
        if angular_momenta[ao2] == ang_mom2:
            cur_coupling_val[:]+=mat[ao2, ao1]*coeffs[ao1]*coeffs[ao2]



@njit(fastmath=True)
def ang_mom_descr(ovlp_mat, coeffs, angular_momenta, atom_ao_range, max_ang_mom, scalar_reps, rho_val):
    scalar_reps[:]=0.0
    for ang_mom in range(1, max_ang_mom+1):
        add_orb_atom_coupling(ovlp_mat, coeffs, angular_momenta, atom_ao_range, 0, ang_mom, ang_mom, scalar_reps[ang_mom-1:ang_mom])
    rho_val[0]=np.sum(scalar_reps)
    add_orb_atom_coupling(ovlp_mat, coeffs, angular_momenta, atom_ao_range, 0, 0, 0, rho_val)

#   Parameters defining how orbitals are represented.

class OML_rep_params:
    def __init__(self, orb_atom_rho_comp=None, max_angular_momentum=3,
                    propagator_coup_mat=False, num_prop_times=1, prop_delta_t=1.0,
                    atom_sorted_pseudo_orbs=False,
                    ofd_coup_mats=False, orb_en_adj=False, ofd_extra_inversions=True):
        self.orb_atom_rho_comp=orb_atom_rho_comp
        self.max_angular_momentum=max_angular_momentum

        self.propagator_coup_mat=propagator_coup_mat
        self.num_prop_times=num_prop_times
        self.prop_delta_t=prop_delta_t

        self.atom_sorted_pseudo_orbs=atom_sorted_pseudo_orbs
        self.ofd_coup_mats=ofd_coup_mats
        self.orb_en_adj=orb_en_adj
        self.ofd_extra_inversions=ofd_extra_inversions

    def __str__(self):
        str_out="orb_atom_rho_comp:"+str(self.orb_atom_rho_comp)+",max_ang_mom:"+str(self.max_angular_momentum)
        if self.propagator_coup_mat:
            str_out+=",prop_delta_t:"+str(self.prop_delta_t)+",num_prop_times:"+str(self.num_prop_times)
        return str_out

def gen_propagator_based_coup_mats(rep_params, hf_orb_coeffs, hf_orb_energies, ovlp_mat):
    naos, num_orbs=hf_orb_coeffs.shape
    inv_hf_orb_coeffs=np.matmul(hf_orb_coeffs.T, ovlp_mat)
    if rep_params.use_Fortran:
        output=np.zeros((naos, naos, rep_params.num_prop_times*2), order='F')
        fgen_ft_coup_mats(inv_hf_orb_coeffs, hf_orb_energies, rep_params.prop_delta_t, naos, num_orbs, rep_params.num_prop_times, output)
        return tuple(np.transpose(output))
    else:
        output=()
        for timestep_counter in range(rep_params.num_prop_times):
            prop_time=(timestep_counter+1)*rep_params.prop_delta_t
            for trigon_func in [math.cos, math.sin]:
                new_mat=np.zeros(num_orbs, dtype=float)
                prop_coeffs=np.array([trigon_func(prop_time*en) for en in hf_orb_energies])
                new_mat=np.matmul(inv_hf_orb_coeffs.T*prop_coeffs, inv_hf_orb_coeffs)
                output=(*output, new_mat)
        return output

def gen_odf_based_coup_mats(rep_params, mo_coeff, mo_energy, mo_occ, ovlp_mat):
    reconstr_mats_kwargs={"mo_energy" : mo_energy, "mo_occ" : mo_occ}
    coupling_matrices=()
    inv_mo_coeff=np.matmul(ovlp_mat, mo_coeff)
    coeff_mats=[inv_mo_coeff, mo_coeff]
    if rep_params.ofd_extra_inversions:
        mat_types_list=[["ovlp", "Fock", "density"] for i in range(2)]
    else:
        mat_types_list=[["ovlp", "Fock"], ["density"]]
    for coeff_mat_id, mat_types in enumerate(mat_types_list):
        added_mats=reconstr_mats(coeff_mats[coeff_mat_id], **reconstr_mats_kwargs,
                                        mat_types=mat_types)
        if len(mat_types)==1:
            coupling_matrices=(*coupling_matrices, added_mats)
        else:
            coupling_matrices=(*coupling_matrices, *added_mats)
    return coupling_matrices

def reconstr_mat(coeff_mat, mo_energy=None, mo_occ=None, mat_type="ovlp"):
    norbs=coeff_mat.shape[1]
    mo_arr=np.ones(norbs, dtype=float)
    for orb_id in range(norbs):
        if mat_type=="density":
            if mo_occ[orb_id]<0.5:
                mo_arr[orb_id]=0.0
        if mat_type=="Fock":
            mo_arr[orb_id]=mo_energy[orb_id]
    return np.matmul(coeff_mat*mo_arr, coeff_mat.T)


def reconstr_mats(mo_coeffs, mo_energy=None, mo_occ=None, mat_types=["ovlp"]):
    output=()
    for mat_type in mat_types:
        output=(*output, reconstr_mat(mo_coeffs, mo_energy=mo_energy, mo_occ=mo_occ, mat_type=mat_type))
    if len(mat_types)==1:
        return output[0]
    else:
        return output

#   Representation of contribution of a single atom to an orb.
class OML_orb_atom_rep:
    def __init__(self, atom_ao_range, coeffs, rep_params, angular_momenta, ovlp_mat):
        self.atom_ao_range=atom_ao_range
        self.scalar_reps=np.zeros(rep_params.max_angular_momentum)
        rho_arr=np.zeros(1)
        ang_mom_descr(ovlp_mat, coeffs, angular_momenta, self.atom_ao_range, rep_params.max_angular_momentum, self.scalar_reps, rho_arr)
        self.rho=rho_arr[0]
        self.pre_renorm_rho=self.rho
        self.atom_id=None
    def completed_scalar_reps(self, coeffs, rep_params, angular_momenta, coup_mats):
        couplings=np.zeros(scalar_coup_length(rep_params, coup_mats))
        gen_orb_atom_scalar_rep(coup_mats, coeffs, angular_momenta, self.atom_ao_range, rep_params.max_angular_momentum, couplings)
        self.scalar_reps=np.append(self.scalar_reps, couplings)
        self.scalar_reps/=self.pre_renorm_rho
        if (rep_params.propagator_coup_mat or rep_params.ofd_coup_mats):
            # TO-DO Do we need this?
            # The parts corresponding to the angular momentum distribution are duplicated, remove:
            self.scalar_reps=self.scalar_reps[rep_params.max_angular_momentum:]
    def energy_readjustment(self, energy_shift, rep_params):
        nam=num_ang_mom(rep_params)
        coup_mat_comp_num=(nam*(3*nam+1))//2
        
        self.scalar_reps[coup_mat_comp_num:2*coup_mat_comp_num]-=self.scalar_reps[:coup_mat_comp_num]*energy_shift
    def __str__(self):
        return "OML_orb_atom_rep,rho:"+str(self.rho)
    def __repr__(self):
        return str(self)

def scalar_coup_length(rep_params, coup_mats):
    nam=num_ang_mom(rep_params)
    return len(coup_mats)*(nam**2+(nam*(nam+1))//2)

def num_ang_mom(rep_params):
    return rep_params.max_angular_momentum+1

#   Auxiliary functions.
def scalar_rep_length(oml_comp):
    try: # if that's a Slater pair list
        first_comp=oml_comp.comps[0]
    except AttributeError: # if that's a compound list
        first_comp=oml_comp
    return len(first_comp.orb_reps[0].orb_atom_reps[0].scalar_reps)

# Generates a representation of what different components of atomic components of orbitals correspond to.
# TO-DO check whether it's correct for numba implementation?
def component_id_ang_mom_map(rep_params):
    output=[]
    if rep_params.propagator_coup_mat:
        # Real and imaginary propagator components for each propagation time plus the overlap matrix.
        num_coup_matrices=rep_params.num_prop_times*2+1
    else:
        # Components corresponding to the angular momentum distribution.
        if rep_params.ofd_coup_mats:
            num_coup_matrices=3
            if rep_params.ofd_extra_inversions:
                num_coup_matrices*=2
        else:
            num_coup_matrices=3 # F, J, and K; or overlap, F, and density
    if not (rep_params.propagator_coup_mat or rep_params.ofd_coup_mats):
        for ang_mom in range(1, num_ang_mom(rep_params)):
            output.append([ang_mom, ang_mom, -1, True])
    for coup_mat_id in range(num_coup_matrices):
        for same_atom in [True, False]:
            for ang_mom1 in range(num_ang_mom(rep_params)):
                for ang_mom2 in range(num_ang_mom(rep_params)):
                    if not (same_atom and (ang_mom1>ang_mom2)):
                        output.append([ang_mom1, ang_mom2, coup_mat_id, same_atom])
    return output

# Auxiliary functions
def generate_atom_ao_ranges(mol):
    ao_sliced_with_shells=mol.aoslice_by_atom()
    output=[]
    for atom_data in ao_sliced_with_shells:
        output.append(atom_data[2:4])
    return np.array(output)

def placeholder_orb_rep(rep_params, atom_ids, atom_ao_ranges, angular_momenta, ovlp_mat, coupling_mats):
    placeholder_orb_coeffs=np.zeros(ovlp_mat.shape[0])
    placeholder_orb_coeffs[0]=1.0
    placeholder_rep=OML_orb_rep(placeholder_orb_coeffs, rep_params, atom_ids, atom_ao_ranges, angular_momenta, ovlp_mat, coupling_mats)
    placeholder_rep.virtual=True
    return placeholder_rep

#   Generate an array of orb representations.
def generate_orb_rep_array(orb_mat, rep_params, aos, atom_ao_ranges, ovlp_mat, coupling_mats):
    atom_ids=[ao.atom_id for ao in aos]
    angular_momenta=np.array([ao.angular for ao in aos])
    if len(orb_mat)==0:
        return [placeholder_orb_rep(rep_params, atom_ids, atom_ao_ranges, angular_momenta, ovlp_mat, coupling_mats)]
    # It's important that ovlp_mat appears first in this array.
    orb_tmat=orb_mat.T
    output=[OML_orb_rep_from_coeffs(orb_coeffs, rep_params, atom_ids, atom_ao_ranges, angular_momenta, ovlp_mat, coupling_mats) for orb_coeffs in orb_tmat]
    return output

def gen_atom_sorted_pseudo_orbs(orb_rep_arr):
    backup_placeholder_arr=[]
    atom_sorted_areps={}
    orb_occ=None
    for orb_rep in orb_rep_arr:
        if orb_rep.virtual:
            backup_placeholder_arr.append(orb_rep)
        else:
            if orb_occ is None:
                orb_occ=orb_rep.rho
            for orb_atom_rep in orb_rep.orb_atom_reps:
                cur_atom_id=orb_atom_rep.atom_id
                if cur_atom_id in atom_sorted_areps:
                    atom_sorted_areps[cur_atom_id].append(orb_atom_rep)
                else:
                    atom_sorted_areps[cur_atom_id]=[orb_atom_rep]
    output=[]
    for atom_id in atom_sorted_areps:
        output.append(OML_orb_rep(None, atom_sorted_areps[atom_id]))
        output[-1].rho=orb_occ
    if len(output)==0:
        return backup_placeholder_arr
    else:
        return output

#   Representation of an orb from atomic contributions.
class OML_orb_rep:
    def __init__(self, orb_coeffs, orb_atom_reps):
        self.rho=0.0
        self.full_coeffs=orb_coeffs
        self.virtual=False
        self.orb_atom_reps=orb_atom_reps
    def add_global_orb_couplings(self, all_orb_coeffs, orb_id, rep_params, atom_ids, angular_momenta, coup_mats):
        global_orb_couplings=np.zeros(len(coup_mats)*((max_angular_momentum+1)*(3*max_angular_momentum+4))/2)
        fgen_orb_global_couplings(all_orb_coeffs, orb_id, angular_momenta, rep_params.max_angular_momentum,
                    len(angular_momenta), coup_mats, coup_mats.shape[2], global_orb_coupling)
        # TO-DO recheck????
        for orb_arep_counter in range(len(self.orb_atom_reps)):
            self.orb_atom_reps[orb_arep_counter].scalar_reps=np.append(self.orb_atom_reps[orb_arep_counter].scalar_reps,
                                        global_orb_couplings)
    def orbital_energy_readjustment(self, Fock_mat, rep_params):
        energy_shift=np.dot(self.full_coeffs, np.matmul(Fock_mat, self.full_coeffs))
        for orb_arep_counter in range(len(self.orb_atom_reps)):
            self.orb_atom_reps[orb_arep_counter].energy_readjustment(energy_shift, rep_params)


def OML_orb_rep_from_coeffs(orb_coeffs, rep_params, atom_ids, atom_ao_ranges, angular_momenta, ovlp_mat, coup_mats):
    atom_list=[]
    prev_atom=-1
    for atom_id, ao_coeff in zip(atom_ids, orb_coeffs):
        if prev_atom != atom_id:
            atom_list.append(atom_id)
            prev_atom=atom_id
    # Each of the resulting groups of AOs is represented with OML_orb_atom_rep object.
    orb_atom_reps=[]
    for atom_id, atom_ao_range in enumerate(atom_ao_ranges):
        cur_orb_atom_rep=OML_orb_atom_rep(atom_ao_range, orb_coeffs, rep_params, angular_momenta, ovlp_mat)
        cur_orb_atom_rep.atom_id=atom_id
        orb_atom_reps.append(cur_orb_atom_rep)
    orb_atom_reps=weighted_array(orb_atom_reps)
    
    orb_atom_reps.normalize_sort_rhos()
    # Try to decrease the number of atomic representations, leaving only the most relevant ones.
    orb_atom_reps.cutoff_minor_weights(remaining_rho=rep_params.orb_atom_rho_comp)
    for orb_arep_counter in range(len(orb_atom_reps)):
        orb_atom_reps[orb_arep_counter].completed_scalar_reps(orb_coeffs, rep_params, angular_momenta, coup_mats)
    return OML_orb_rep(orb_coeffs, orb_atom_reps)

class weighted_array(list):
    def normalize_rhos(self, normalization_constant=None):
        if normalization_constant is None:
            normalization_constant=sum(el.rho for el in self)
        for i in range(len(self)):
            self[i].rho/=normalization_constant
    def sort_rhos(self):
        self.sort(key=lambda x: x.rho, reverse=True)
    def normalize_sort_rhos(self):
        self.normalize_rhos()
        self.sort_rhos()
    def cutoff_minor_weights(self, remaining_rho=None):
        if (remaining_rho is not None) and (len(self)>1):
            ignored_rhos=0.0
            for remaining_length in range(len(self),0,-1):
                upper_cutoff=self[remaining_length-1].rho
                cut_rho=upper_cutoff*remaining_length+ignored_rhos
                if cut_rho>(1.0-remaining_rho):
                    density_cut=(1.0-remaining_rho-ignored_rhos)/remaining_length
                    break
                else:
                    ignored_rhos+=upper_cutoff
            del(self[remaining_length:])
            for el_id in range(remaining_length):
                self[el_id].rho=max(0.0, self[el_id].rho-density_cut) # max was introduced in case there is some weird numerical noise.
            self.normalize_rhos()


