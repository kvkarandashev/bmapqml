from numba import njit, prange
import numpy as np
import itertools, copy

import datetime


# For timestamps.
def now():
    return datetime.datetime.now()


# Whether we are looking at a Slater determinant or a compound.
def is_pair_reps(comp_arr):
    return hasattr(comp_arr[0], "comps")


# Iterate over representations of orbitals.
def iterated_orb_reps(oml_comp, pair_reps=False, single_orb_list=False):
    if pair_reps:
        return itertools.chain(oml_comp.comps[0].orb_reps, oml_comp.comps[1].orb_reps)
    else:
        if single_orb_list:
            return [oml_comp]
        else:
            return oml_comp.orb_reps


def orb_rep_rho_list(oml_comp, pair_reps=False):
    output = []
    for orb in iterated_orb_reps(oml_comp, pair_reps=pair_reps):
        output.append([orb.rho, orb])
    if pair_reps:
        for i in range(len(oml_comp.comps[0].orb_reps)):
            output[i][0] *= -1
    return output


# Basic dot product expressions
# Orbital-orbital product with derivatives.
@njit(fastmath=True)
def make_sqdiffs_arr(vec1, vec2, sqdiffs_arr):
    sqdiffs_arr[:] = vec1[:]
    sqdiffs_arr -= vec2
    sqdiffs_arr **= 2


@njit(fastmath=True)
def orb_orb_cov_wders(
    orb_areps_A,
    orb_areps_B,
    arep_rhos_A,
    arep_rhos_B,
    num_orb_areps_A,
    num_orb_areps_B,
    sqdiffs_arr,
    inout_arr,
):
    inout_arr[:] = 0.0
    for arep_A_id in range(num_orb_areps_A):
        rho_A = arep_rhos_A[arep_A_id]
        for arep_B_id in range(num_orb_areps_B):
            rho_B = arep_rhos_B[arep_B_id]
            make_sqdiffs_arr(
                orb_areps_A[arep_A_id, :], orb_areps_B[arep_B_id, :], sqdiffs_arr
            )
            orb_comp = rho_A * rho_B * np.exp(-np.sum(sqdiffs_arr))
            inout_arr[0] += orb_comp
            inout_arr[1:] += orb_comp * sqdiffs_arr


# Orbital-orbital product.
@njit(fastmath=True)
def orb_orb_cov(
    orb_areps_A,
    orb_areps_B,
    arep_rhos_A,
    arep_rhos_B,
    num_orb_areps_A,
    num_orb_areps_B,
    sqdiffs_arr,
    inout_arr,
):
    inout_arr[:] = 0.0
    for arep_A_id in range(num_orb_areps_A):
        rho_A = arep_rhos_A[arep_A_id]
        for arep_B_id in range(num_orb_areps_B):
            rho_B = arep_rhos_B[arep_B_id]
            make_sqdiffs_arr(
                orb_areps_A[arep_A_id, :], orb_areps_B[arep_B_id, :], sqdiffs_arr
            )
            inout_arr[0] += rho_A * rho_B * np.exp(-np.sum(sqdiffs_arr))


# Molecule-molecule dot product
@njit(fastmath=True)
def lin_sep_orb_kernel_el(
    A_orb_areps,
    B_orb_areps,
    A_arep_rhos,
    B_arep_rhos,
    A_orb_rhos,
    B_orb_rhos,
    A_orb_atom_nums,
    B_orb_atom_nums,
    A_orb_num,
    B_orb_num,
    sqdiffs_arr,
    orb_temp_arr,
    inout_arr,
):
    inout_arr[:] = 0.0
    for A_orb_id in range(A_orb_num):
        rho_A = A_orb_rhos[A_orb_id]
        for B_orb_id in range(B_orb_num):
            orb_orb_cov(
                A_orb_areps[A_orb_id, :, :],
                B_orb_areps[B_orb_id, :, :],
                A_arep_rhos[A_orb_id, :],
                B_arep_rhos[B_orb_id, :],
                A_orb_atom_nums[A_orb_id],
                B_orb_atom_nums[B_orb_id],
                sqdiffs_arr,
                orb_temp_arr,
            )
            inout_arr += rho_A * B_orb_rhos[B_orb_id] * orb_temp_arr


# Molecule-molecule dot product with derivatives.
@njit(fastmath=True)
def lin_sep_orb_kernel_el_wders(
    A_orb_areps,
    B_orb_areps,
    A_arep_rhos,
    B_arep_rhos,
    A_orb_rhos,
    B_orb_rhos,
    A_sp_log_ders,
    B_sp_log_ders,
    A_orb_atom_nums,
    B_orb_atom_nums,
    A_orb_num,
    B_orb_num,
    sqdiffs_arr,
    orb_temp_arr,
    inout_arr,
):
    inout_arr[:] = 0.0
    for A_orb_id in range(A_orb_num):
        rho_A = A_orb_rhos[A_orb_id]
        for B_orb_id in range(B_orb_num):
            orb_orb_cov_wders_log_incl(
                A_orb_areps[A_orb_id],
                B_orb_areps[B_orb_id],
                A_arep_rhos[A_orb_id],
                B_arep_rhos[B_orb_id],
                A_sp_log_ders[A_orb_id],
                B_sp_log_ders[B_orb_id],
                A_orb_atom_nums[A_orb_id],
                B_orb_atom_nums[B_orb_id],
                sqdiffs_arr,
                orb_temp_arr,
            )
            inout_arr += rho_A * B_orb_rhos[B_orb_id] * orb_temp_arr


# Self-covariance (used for normalizing the orbitals).
@njit(fastmath=True)
def orb_self_cov(orb_areps, arep_rhos, num_orb_areps, sqdiffs_arr, inout_arr):
    orb_orb_cov(
        orb_areps,
        orb_areps,
        arep_rhos,
        arep_rhos,
        num_orb_areps,
        num_orb_areps,
        sqdiffs_arr,
        inout_arr,
    )


@njit(fastmath=True)
def orb_self_cov_wders(orb_areps, arep_rhos, orb_atom_nums, sqdiffs_arr, inout_arr):
    orb_orb_cov_wders(
        orb_areps,
        orb_areps,
        arep_rhos,
        arep_rhos,
        orb_atom_nums,
        orb_atom_nums,
        sqdiffs_arr,
        inout_arr,
    )


@njit(fastmath=True)
def mol_self_cov(
    orb_areps,
    arep_rhos,
    orb_rhos,
    orb_atom_nums,
    orb_num,
    sqdiffs_arr,
    orb_temp_arr,
    inout_arr,
):
    lin_sep_orb_kernel_el(
        orb_areps,
        orb_areps,
        arep_rhos,
        arep_rhos,
        orb_rhos,
        orb_rhos,
        orb_atom_nums,
        orb_atom_nums,
        orb_num,
        orb_num,
        sqdiffs_arr,
        orb_temp_arr,
        inout_arr,
    )


@njit(fastmath=True)
def orb_orb_cov_wders_log_incl(
    orb_areps_A,
    orb_areps_B,
    arep_rhos_A,
    arep_rhos_B,
    A_sp_log_ders,
    B_sp_log_ders,
    num_orb_areps_A,
    num_orb_areps_B,
    sqdiffs_arr,
    inout_arr,
):
    orb_orb_cov_wders(
        orb_areps_A,
        orb_areps_B,
        arep_rhos_A,
        arep_rhos_B,
        num_orb_areps_A,
        num_orb_areps_B,
        sqdiffs_arr,
        inout_arr,
    )
    inout_arr[1:] -= inout_arr[0] * (A_sp_log_ders + B_sp_log_ders) / 2


@njit(fastmath=True)
def mol_self_cov_wders(
    orb_areps,
    arep_rhos,
    orb_rhos,
    sp_log_ders,
    orb_atom_nums,
    orb_num,
    sqdiffs_arr,
    orb_temp_arr,
    inout_arr,
):
    lin_sep_orb_kernel_el_wders(
        orb_areps,
        orb_areps,
        arep_rhos,
        arep_rhos,
        orb_rhos,
        orb_rhos,
        sp_log_ders,
        sp_log_ders,
        orb_atom_nums,
        orb_atom_nums,
        orb_num,
        orb_num,
        sqdiffs_arr,
        orb_temp_arr,
        inout_arr,
    )


# Input for Gausiann Molecular Orbitals kernel.
# "sep" - separated - is to distinguish from other kernels that didn't make the final cut.
class GMO_sep_orb_kern_input:
    def __init__(self, oml_compound_array=None, pair_reps=None):
        if pair_reps is None:
            pair_reps = is_pair_reps(oml_compound_array)
        if pair_reps:
            self.max_num_scalar_reps = len(
                oml_compound_array[0].comps[0].orb_reps[0].orb_atom_reps[0].scalar_reps
            )
        else:
            self.max_num_scalar_reps = len(
                oml_compound_array[0].orb_reps[0].orb_atom_reps[0].scalar_reps
            )

        self.num_mols = len(oml_compound_array)
        # Number of orbitals in a molecule
        self.orb_nums = np.zeros((self.num_mols,), dtype=int)
        # Maximum number of orbitals.
        self.max_num_orbs = 0

        for comp_id, oml_comp in enumerate(oml_compound_array):
            rho_orb_list = orb_rep_rho_list(oml_comp, pair_reps=pair_reps)
            self.orb_nums[comp_id] = len(rho_orb_list)

        self.max_num_orbs = max(self.orb_nums)
        # Number of atom contributions included in a given orbital.
        self.orb_atom_nums = np.zeros((self.num_mols, self.max_num_orbs), dtype=int)
        # Weights associated with orbitals.
        self.orb_rhos = np.zeros((self.num_mols, self.max_num_orbs))
        for comp_id, oml_comp in enumerate(oml_compound_array):
            rho_orb_list = orb_rep_rho_list(oml_comp, pair_reps=pair_reps)
            for orb_id, [orb_rho, orb_rep] in enumerate(rho_orb_list):
                self.orb_atom_nums[comp_id, orb_id] = len(orb_rep.orb_atom_reps)
                self.orb_rhos[comp_id, orb_id] = orb_rho

        # Maximum number of atoms in an orbital
        self.max_num_orb_atom_reps = np.amax(self.orb_atom_nums)

        # Weights of atomic contributions in an orbital.
        self.orb_arep_rhos = np.zeros(
            (self.num_mols, self.max_num_orbs, self.max_num_orb_atom_reps)
        )
        # Representations of atomic contributions to orbitals.
        self.orb_atom_sreps = np.zeros(
            (
                self.num_mols,
                self.max_num_orbs,
                self.max_num_orb_atom_reps,
                self.max_num_scalar_reps,
            )
        )
        for ind_comp, oml_comp in enumerate(oml_compound_array):
            for ind_orb, [orb_rho, orb_rep] in enumerate(
                orb_rep_rho_list(oml_comp, pair_reps=pair_reps)
            ):
                for ind_orb_arep, orb_arep in enumerate(orb_rep.orb_atom_reps):
                    self.orb_arep_rhos[ind_comp, ind_orb, ind_orb_arep] = orb_arep.rho
                    self.orb_atom_sreps[
                        ind_comp, ind_orb, ind_orb_arep, :
                    ] = orb_arep.scalar_reps[:]
        # Representations of atomic contributions scaled by sigma parameters.
        self.orb_atom_scaled_sreps = None

    # Generate atomic contribution representations scaled by sigmas.
    @staticmethod
    @njit(fastmath=True)
    def width_rescaling(orb_atom_sreps, orb_atom_nums, orb_nums, sigmas):
        orb_atom_scaled_sreps = np.copy(orb_atom_sreps)
        num_mols = orb_atom_scaled_sreps.shape[0]
        for mol_id in range(num_mols):
            for orb_id in range(orb_nums[mol_id]):
                for arep_id in range(orb_atom_nums[mol_id, orb_id]):
                    orb_atom_scaled_sreps[mol_id, orb_id, arep_id, :] /= 2 * sigmas
        return orb_atom_scaled_sreps

    # Renormalize arep rho coefficients to make scalar product between orbitals unity.
    @staticmethod
    @njit(fastmath=True)
    def numba_lin_sep_kern_renormalize_arep_rhos(
        orb_atom_scaled_sreps, orb_nums, orb_atom_nums, orb_arep_rhos
    ):
        num_mols = orb_atom_scaled_sreps.shape[0]

        for mol_id in range(num_mols):
            sqdiffs_arr = np.zeros((orb_atom_scaled_sreps.shape[-1],))
            temp_arr = np.zeros((1,))
            for orb_id in range(orb_nums[mol_id]):
                orb_self_cov(
                    orb_atom_scaled_sreps[mol_id, orb_id, :, :],
                    orb_arep_rhos[mol_id, orb_id, :],
                    orb_atom_nums[mol_id, orb_id],
                    sqdiffs_arr,
                    temp_arr,
                )
                orb_arep_rhos[mol_id, orb_id, :] /= np.sqrt(temp_arr[0])
        return orb_arep_rhos

    # Rescale atomic contributions by sigmas.
    def rescale_reps(self, sigmas):
        self.orb_atom_scaled_sreps = self.width_rescaling(
            self.orb_atom_sreps, self.orb_atom_nums, self.orb_nums, sigmas
        )

    # Generate normalization coefficients for orbitals and the corresponding derivatives w.r.t. sigmas.
    @staticmethod
    @njit(fastmath=True)
    def orb_self_product_log_ders(orb_areps, arep_rhos, orb_atom_nums, orb_nums):
        num_mols = orb_areps.shape[0]
        max_num_orbs = orb_areps.shape[1]

        num_ders = orb_areps.shape[-1]
        orb_comp_dim = num_ders + 1

        log_ders = np.zeros((num_mols, max_num_orbs, num_ders))

        for mol_id in range(num_mols):
            # TO-DO is there a Fortran-like way to define sqdiffs_arr and orb_temp_arr outside the loop and make them private??
            sqdiffs_arr = np.zeros((num_ders,))
            orb_temp_arr = np.zeros((orb_comp_dim,))
            for orb_id in range(orb_nums[mol_id]):
                orb_self_cov_wders(
                    orb_areps[mol_id, orb_id, :, :],
                    arep_rhos[mol_id, orb_id, :],
                    orb_atom_nums[mol_id, orb_id],
                    sqdiffs_arr,
                    orb_temp_arr,
                )
                log_ders[mol_id, orb_id, :] = orb_temp_arr[1:] / orb_temp_arr[0]

        return log_ders

    # Calculate normalization coefficients for molecules.
    @staticmethod
    @njit(fastmath=True)
    def calc_mol_norm_constants(
        orb_areps, arep_rhos, orb_rhos, orb_atom_nums, orb_nums
    ):
        num_mols = orb_areps.shape[0]
        output = np.zeros((num_mols,))

        for mol_id in range(num_mols):
            sqdiffs_arr = np.zeros((orb_areps.shape[-1],))
            orb_temp_arr = np.zeros((1,))
            mol_self_cov(
                orb_areps[mol_id, :, :, :],
                arep_rhos[mol_id, :, :],
                orb_rhos[mol_id, :],
                orb_atom_nums[mol_id, :],
                orb_nums[mol_id],
                sqdiffs_arr,
                orb_temp_arr,
                output[mol_id : mol_id + 1],
            )
            output[mol_id] = np.sqrt(output[mol_id]) ** (-1)
        return output

    # Calculate normalization coefficients for molecules and the corresponding derivatives w.r.t. sigmas.
    @staticmethod
    @njit(fastmath=True)
    def calc_mol_norm_constants_wders(
        orb_areps,
        arep_rhos,
        orb_rhos,
        orb_sp_log_ders,
        orb_atom_nums,
        orb_nums,
        der_resc,
    ):
        num_mols = orb_areps.shape[0]
        kern_comp_dim = orb_areps.shape[-1] + 1
        output = np.zeros((num_mols, kern_comp_dim))

        for mol_id in range(num_mols):
            sqdiffs_arr = np.zeros((orb_areps.shape[-1],))
            orb_temp_arr = np.zeros((kern_comp_dim,))
            mol_self_cov_wders(
                orb_areps[mol_id],
                arep_rhos[mol_id],
                orb_rhos[mol_id],
                orb_sp_log_ders[mol_id],
                orb_atom_nums[mol_id],
                orb_nums[mol_id],
                sqdiffs_arr,
                orb_temp_arr,
                output[mol_id],
            )
            output[mol_id, 1:] /= 2 * output[mol_id, 0]
            # TODO doing the rescaling here is illogical
            output[mol_id, 1:] *= der_resc
            output[mol_id, 0] = np.sqrt(output[mol_id, 0]) ** (-1)
        return output

    # TODO: The name is misleading.
    # Generate intermediate data used in kernel calculations,
    # namely rescaled representations, normalization constants for orbitals and molecules,
    # and derivatives of the latter.
    def lin_sep_kern_renormalize_arep_rhos(
        self, sigmas, with_ders=False, mol_lin_norm=False
    ):
        self.rescale_reps(sigmas)
        self.numba_lin_sep_kern_renormalize_arep_rhos(
            self.orb_atom_scaled_sreps,
            self.orb_nums,
            self.orb_atom_nums,
            self.orb_arep_rhos,
        )
        if with_ders:
            self.orb_sp_log_ders = self.orb_self_product_log_ders(
                self.orb_atom_scaled_sreps,
                self.orb_arep_rhos,
                self.orb_atom_nums,
                self.orb_nums,
            )
        if mol_lin_norm:
            if with_ders:
                self.mol_norm_constants = self.calc_mol_norm_constants_wders(
                    self.orb_atom_scaled_sreps,
                    self.orb_arep_rhos,
                    self.orb_rhos,
                    self.orb_sp_log_ders,
                    self.orb_atom_nums,
                    self.orb_nums,
                    2.0 / sigmas,
                )
            else:
                self.mol_norm_constants = self.calc_mol_norm_constants(
                    self.orb_atom_scaled_sreps,
                    self.orb_arep_rhos,
                    self.orb_rhos,
                    self.orb_atom_nums,
                    self.orb_nums,
                )

    def Gauss_sep_orb_kernel_args(self):
        return (
            self.orb_atom_scaled_sreps,
            self.orb_arep_rhos,
            self.orb_rhos,
            self.orb_sp_log_ders,
            self.orb_atom_nums,
            self.orb_nums,
            self.mol_norm_constants,
        )


@njit(fastmath=True)
def numba_global_Gauss_sep_orb_kernel_wders(
    A_orb_areps,
    A_arep_rhos,
    A_orb_rhos,
    A_sp_log_ders,
    A_orb_atom_nums,
    A_orb_nums,
    A_mol_norms,
    B_orb_areps,
    B_arep_rhos,
    B_orb_rhos,
    B_sp_log_ders,
    B_orb_atom_nums,
    B_orb_nums,
    B_mol_norms,
    kern_der_resc,
    inv_sq_sigma,
):
    A_num_mols = A_orb_areps.shape[0]

    B_num_mols = B_orb_areps.shape[0]

    kern_comp_dim = A_orb_areps.shape[-1] + 1
    Kernel = np.zeros((A_num_mols, B_num_mols, kern_comp_dim + 1))

    mol_temp_arr = np.empty(
        (kern_comp_dim,),
    )
    orb_temp_arr = np.empty(
        (kern_comp_dim,),
    )

    for A_mol_id in range(A_num_mols):
        for B_mol_id in range(B_num_mols):
            # Calculate kernel element between molecules with indices A_mol_id and B_mol_id

            # Sum contributions from orbitals in A_mol_id and B_mol_id
            for A_orb_id in range(A_orb_nums[A_mol_id]):
                for B_orb_id in range(B_orb_nums[B_mol_id]):
                    mol_temp_arr[:] = 0.0
                    # Sum contributions from different atoms.
                    for A_arep_id in range(A_orb_atom_nums[A_mol_id, A_orb_id]):
                        for B_arep_id in range(B_orb_atom_nums[B_mol_id, B_orb_id]):
                            # Dot product and its derivatives.
                            orb_temp_arr[1:] = A_orb_areps[
                                A_mol_id, A_orb_id, A_arep_id, :
                            ]
                            orb_temp_arr[1:] -= B_orb_areps[
                                B_mol_id, B_orb_id, B_arep_id, :
                            ]
                            orb_temp_arr[1:] **= 2
                            orb_temp_arr[0] = np.exp(-np.sum(orb_temp_arr[1:]))
                            orb_temp_arr[1:] *= orb_temp_arr[0]
                            orb_temp_arr[:] *= (
                                A_arep_rhos[A_mol_id, A_orb_id, A_arep_id]
                                * B_arep_rhos[B_mol_id, B_orb_id, B_arep_id]
                            )
                            mol_temp_arr[:] += orb_temp_arr[:]
                    # Correct derivatives by factors caused by normalization.
                    mol_temp_arr[1:] -= (
                        mol_temp_arr[0]
                        * (
                            A_sp_log_ders[A_mol_id, A_orb_id, :]
                            + B_sp_log_ders[B_mol_id, B_orb_id, :]
                        )
                        / 2
                    )
                    # Multiply by weights of orbitals.
                    mol_temp_arr[:] *= (
                        A_orb_rhos[A_mol_id, A_orb_id] * B_orb_rhos[B_mol_id, B_orb_id]
                    )

                    # Add the summed contribution to the kernel element.
                    Kernel[A_mol_id, B_mol_id, 1:] += mol_temp_arr[:]

            # With orbital contributions summed, we convert linear kernel element to Gaussian one.
            # Rescale some derivatives
            Kernel[A_mol_id, B_mol_id, 2:] *= kern_der_resc[1:]
            # Account for normalization in product value.
            Kernel[A_mol_id, B_mol_id, 2:] /= Kernel[A_mol_id, B_mol_id, 1]
            # Account for normalization in derivatives.
            Kernel[A_mol_id, B_mol_id, 2:] -= (
                A_mol_norms[A_mol_id, 1:] + B_mol_norms[B_mol_id, 1:]
            )

            Kernel[A_mol_id, B_mol_id, 1] *= (
                A_mol_norms[A_mol_id, 0] * B_mol_norms[B_mol_id, 0]
            )
            Kernel[A_mol_id, B_mol_id, 2:] *= Kernel[A_mol_id, B_mol_id, 1]
            # Final coversion to Gaussian
            Kernel[A_mol_id, B_mol_id, 0] = np.exp(
                -(1.0 - Kernel[A_mol_id, B_mol_id, 1]) * inv_sq_sigma
            )
            # Derivative with respect to Gaussian sigma.
            Kernel[A_mol_id, B_mol_id, 1] = Kernel[A_mol_id, B_mol_id, 0] * (
                1 - Kernel[A_mol_id, B_mol_id, 1]
            )
            # Other derivatives
            Kernel[A_mol_id, B_mol_id, 2:] *= (
                inv_sq_sigma * Kernel[A_mol_id, B_mol_id, 0]
            )

            Kernel[A_mol_id, B_mol_id, 1] *= kern_der_resc[0]
    return Kernel


def global_Gauss_sep_orb_kernel_conv(Ac, Bc, sigmas):
    inv_sq_sigma = 1.0 / sigmas[0] ** 2
    kern_der_resc = np.array([2.0 / sigmas[0] ** 3, *2.0 / sigmas[1:]])
    return numba_global_Gauss_sep_orb_kernel_wders(
        *Ac.Gauss_sep_orb_kernel_args(),
        *Bc.Gauss_sep_orb_kernel_args(),
        kern_der_resc,
        inv_sq_sigma,
    )


def gauss_sep_orb_kernel(A, B, sigmas, with_ders=False, global_Gauss=True):
    Ac = GMO_sep_orb_kern_input(oml_compound_array=A)
    Bc = GMO_sep_orb_kern_input(oml_compound_array=B)
    aux_data_start = now()
    Ac.lin_sep_kern_renormalize_arep_rhos(sigmas[1:], with_ders=True, mol_lin_norm=True)
    Bc.lin_sep_kern_renormalize_arep_rhos(sigmas[1:], with_ders=True, mol_lin_norm=True)
    aux_data_finish = now()
    print("Auxiliary data generation", aux_data_finish - aux_data_start)
    output = global_Gauss_sep_orb_kernel_conv(Ac, Bc, sigmas)
    print("numba_global_Guass_sep_orb_kernel_wders call:", now() - aux_data_finish)
    return output


def gauss_sep_orb_sym_kernel(A, sigmas, with_ders=False, global_Gauss=True):
    Ac = GMO_sep_orb_kern_input(oml_compound_array=A)
    aux_data_start = now()
    Ac.lin_sep_kern_renormalize_arep_rhos(sigmas[1:], with_ders=True, mol_lin_norm=True)
    aux_data_finish = now()
    print("Auxiliary data generation", aux_data_finish - aux_data_start)
    output = global_Gauss_sep_orb_kernel_conv(Ac, Ac, sigmas)
    print("numba_global_Guass_sep_orb_kernel_wders call:", now() - aux_data_finish)
    return output
