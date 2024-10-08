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

# TO-DO change the way the default spin is chosen?

import numpy as np
import copy
from pyscf import lo, gto, scf, dft
from os.path import isfile
from .representations import (
    generate_orb_rep_array,
    gen_propagator_based_coup_mats,
    gen_atom_sorted_pseudo_orbs,
    gen_odf_based_coup_mats,
    generate_atom_ao_ranges,
    generate_ao_arr,
)
from ..utils import dump2pkl, loadpkl, OptionUnavailableError, read_xyz_file

# WARNING: THIS INTERFACE HAS PROBLEMS
from .xtb_interface import generate_pyscf_mf_mol as xtb_generate_pyscf_mf_mol
from .tblite_interface import generate_pyscf_mf_mol as tblite_generate_pyscf_mf_mol

mf_creator = {"HF": scf.RHF, "UHF": scf.UHF, "KS": dft.RKS, "UKS": dft.UKS}

unrestricted_methods = ["UHF", "UKS"]

HF_methods = ["HF", "UHF"]

KS_methods = ["KS", "UKS"]

neglect_orb_occ = 0.1

#   TO-DO rename to "available_orbital_types" now that Boys is also used?
available_orb_types = ["standard", "HOMO_removed", "LUMO_added", "first_excitation"]

available_localization_procedures = ["Boys", "IBO", "Pipek-Mezey", None]

available_software = ["pySCF", "molpro", "xTB", "tblite"]

available_calc_types = {
    "pySCF": ["HF", "UHF", "KS", "UKS"],
    "xTB": ["xTB"],
    "tblite": ["GFN2-xTB", "GFN1-xTB"],
    "molpro": ["HF", "UHF"],
}


def assign_avail_check(val, avail_val):
    if val in avail_val:
        return val
    else:
        print("Not found:", val, avail_val)
        raise OptionUnavailableError


def pySCFNotConvergedErrorMessage(oml_comp=None):
    if oml_comp is None:
        message = "WARNING: A SCF calculation failed to converge."
    else:
        message = (
            "WARNING: A SCF calculation failed to converge at "
            + str(oml_comp.pyscf_calc_params.scf_max_cycle)
            + " cycles."
        )
        if oml_comp.mats_savefile is not None:
            message += " Problematic mats_savefile: " + oml_comp.mats_savefile
    return message


class pySCFNotConvergedError(Exception):
    pass


class OML_compound:
    """'Orbital Machine Learning (OML) compound' class is used to store data normally
    part of the compound class along with results of ab initio calculations done with
    the pySCF package.

    xyz             - xyz file use to create base Compound object.
    mats_savefile   - the file where results of the ab initio calculations are stored;
                      if it is the name is specified then the results would be imported from the file if it exists
                      or saved to the file otherwise.
    calc_type       - type of the calculation (for now only HF with orb localization and the default basis set are supported).
    """

    def __init__(
        self,
        xyz=None,
        coordinates=None,
        nuclear_charges=None,
        atomtypes=None,
        mats_savefile=None,
        calc_type=None,
        basis="sto-3g",
        used_orb_type="standard",
        use_Huckel=False,
        optimize_geometry=False,
        charge=0,
        spin=None,
        dft_xc="lda,vwn",
        dft_nlc="",
        software="pySCF",
        pyscf_calc_params=None,
        use_pyscf_localization=True,
        write_full_pyscf_chkfile=False,
        solvent_eps=None,
        localization_procedure="IBO",
        temp_calc_dir=None,
    ):
        self.coordinates = None
        self.nuclear_charges = None
        self.atomtypes = None
        if xyz is not None:
            (
                self.nuclear_charges,
                self.atomtypes,
                self.coordinates,
                self.add_attr_dict,
            ) = read_xyz_file(xyz)
        if coordinates is not None:
            self.coordinates = coordinates
        if nuclear_charges is not None:
            self.nuclear_charges = nuclear_charges
        if atomtypes is not None:
            self.atomtypes = atomtypes
            self.nuclear_charges = np.array(
                [nuclear_charge(atomtype) for atomtype in self.atomtypes], dtype=int
            )

        self.software = assign_avail_check(software, available_software)

        if calc_type is None:
            self.calc_type = available_calc_types[self.software][0]
        else:
            self.calc_type = assign_avail_check(
                calc_type, available_calc_types[self.software]
            )

        self.charge = charge

        if spin is None:
            self.spin = self.default_spin_val()
        else:
            self.spin = spin

        self.mats_savefile = mats_savefile
        self.basis = basis
        self.used_orb_type = assign_avail_check(used_orb_type, available_orb_types)
        self.use_Huckel = use_Huckel
        self.optimize_geometry = optimize_geometry
        if self.software == "xTB":
            self.calc_type = "xTB"
        self.use_pyscf_localization = use_pyscf_localization
        self.solvent_eps = solvent_eps
        self.localization_procedure = localization_procedure

        self.temp_calc_dir = temp_calc_dir

        self.pyscf_chkfile = None
        self.full_pyscf_chkfile = None

        self.write_full_pyscf_chkfile = write_full_pyscf_chkfile

        if pyscf_calc_params is None:
            self.pyscf_calc_params = OML_pyscf_calc_params()
        else:
            self.pyscf_calc_params = pyscf_calc_params
        self.dft_xc = dft_xc
        self.dft_nlc = dft_nlc

        self.mats_created = None

        self.mo_coeff = None
        self.mo_occ = None
        self.mo_energy = None
        self.aos = None
        self.atom_ao_ranges = None
        self.e_tot = None

        self.ovlp_mat = None
        self.orb_mat = None
        if self.calc_type in HF_methods:
            self.j_mat = None
            self.k_mat = None
            self.fock_mat = None
        if self.optimize_geometry:
            self.opt_coords = None

        self.orb_reps = []

    # Parameters of the reference ab initio calculations.
    def calc_params(self):
        return {
            "calc_type": self.calc_type,
            "basis": self.basis,
            "software": self.software,
            "use_pyscf_localization": self.use_pyscf_localization,
            "used_orb_type": self.used_orb_type,
            "use_Huckel": self.use_Huckel,
            "pyscf_calc_params": self.pyscf_calc_params,
            "use_pyscf_localization": self.use_pyscf_localization,
            "solvent_eps": self.solvent_eps,
            "localization_procedure": self.localization_procedure,
            "dft_xc": self.dft_xc,
            "dft_nlc": self.dft_nlc,
        }

    def default_spin_val(self):
        return (sum(self.nuclear_charges) - self.charge) % 2

    def assign_calc_res(self, calc_res_dict):
        self.mo_coeff = calc_res_dict["mo_coeff"]
        self.mo_occ = calc_res_dict["mo_occ"]
        self.mo_energy = calc_res_dict["mo_energy"]
        self.aos = calc_res_dict["aos"]
        self.atom_ao_ranges = calc_res_dict["atom_ao_ranges"]
        self.e_tot = calc_res_dict["e_tot"]

        self.ovlp_mat = calc_res_dict["ovlp_mat"]
        self.orb_mat = calc_res_dict["orb_mat"]

        if self.calc_type in HF_methods:
            self.j_mat = calc_res_dict["j_mat"]
            self.k_mat = calc_res_dict["k_mat"]
            self.fock_mat = calc_res_dict["fock_mat"]
        if self.optimize_geometry:
            self.opt_coords = calc_res_dict["opt_coords"]

    def check_saved_files(self):
        if self.mats_savefile is not None:
            if self.mats_savefile.endswith(".pkl"):
                self.pyscf_chkfile = self.mats_savefile[:-3] + "chkfile"
            else:
                savefile_prename = (
                    self.mats_savefile + "." + self.calc_type + "." + self.basis
                )
                if self.use_Huckel:
                    savefile_prename += ".Huckel"
                if self.optimize_geometry:
                    savefile_prename += ".geom_opt"
                if self.charge != 0:
                    savefile_prename += ".charge_" + str(self.charge)
                if self.spin != self.default_spin_val():
                    savefile_prename += ".spin_" + str(self.spin)
                if self.calc_type in KS_methods:
                    savefile_prename += (
                        ".xc_" + self.dft_xc + ".nlc_" + str(self.dft_nlc)
                    )
                if self.software != "pySCF":
                    savefile_prename += (
                        "."
                        + self.software
                        + ".pySCF_loc_"
                        + str(self.use_pyscf_localization)
                    )
                if self.solvent_eps is not None:
                    savefile_prename += ".solvent_eps_" + str(self.solvent_eps)
                self.pyscf_chkfile = savefile_prename + ".chkfile"
                self.mats_savefile = savefile_prename + "." + self.used_orb_type
                if self.localization_procedure is not None:
                    self.mats_savefile += ".localization_" + self.localization_procedure
                self.mats_savefile += ".pkl"
            if self.write_full_pyscf_chkfile:
                self.full_pyscf_chkfile = self.pyscf_chkfile + "_full"
        self.mats_created = ext_isfile(self.mats_savefile)
        if self.mats_created:
            # Import ab initio results from the savefile.
            precalc_vals = loadpkl(self.mats_savefile)
            self.assign_calc_res(precalc_vals)

    def run_calcs(self, initial_guess_comp=None):
        """Runs the ab initio calculations if they are necessary.

        pyscf_calc_params   - object of OML_pyscf_calc_params class containing parameters of the pySCF calculation. (To be made more useful.)
        """
        self.check_saved_files()
        if not self.mats_created:
            if self.software == "molpro":
                if self.used_orb_type != "standard_orb":
                    raise OptionUnavailableError
                from .molpro_interface import get_molpro_calc_data

                calc_data = get_molpro_calc_data[self.calc_type](self)
                self.assign_calc_res(calc_data)
                # TO-DO IMPORT BASIS FROM MOLPRO BEFORE LOCALIZATION???
                if self.use_pyscf_localization:
                    self.orb_mat = self.localized_orbs()
                self.create_mats_savefile()
                return
            # Run the pySCF calculations.
            mf, pyscf_mol = self.generate_pyscf_mf_mol(
                initial_guess_comp=initial_guess_comp
            )
            ### Special operations.
            if self.used_orb_type != "standard_orb":
                self.mo_occ = self.alter_mo_occ(mf.mo_occ)
            ###
            self.mo_coeff = self.adjust_spin_mat_dimen(mf.mo_coeff)
            self.mo_occ = self.adjust_spin_mat_dimen(mf.mo_occ)
            self.mo_energy = self.adjust_spin_mat_dimen(mf.mo_energy)
            self.aos = generate_ao_arr(pyscf_mol)
            self.atom_ao_ranges = generate_atom_ao_ranges(pyscf_mol)
            self.e_tot = mf.e_tot
            self.ovlp_mat = pyscf_mol.intor_symmetric("int1e_ovlp")

            if (self.calc_type in HF_methods) and (self.solvent_eps is None):
                self.j_mat = self.adjust_spin_mat_dimen(mf.get_j())
                self.k_mat = self.adjust_spin_mat_dimen(mf.get_k())
                self.fock_mat = self.adjust_spin_mat_dimen(mf.get_fock())

            self.orb_mat = self.localized_orbs(pyscf_mol=pyscf_mol)
            self.create_mats_savefile()

    def localized_orbs(self, pyscf_mol=None):
        occ_orb_arrs = self.occ_orbs()
        if pyscf_mol is None:
            pyscf_mol = self.generate_pyscf_mol()
        orb_mat = []
        for occ_orb_arr in occ_orb_arrs:
            if occ_orb_arr.size == 0:
                orb_mat.append([])
            else:
                if self.localization_procedure not in available_localization_procedures:
                    raise OptionUnavailableError
                if self.localization_procedure is None:
                    new_orb_mat = occ_orb_arr
                elif self.localization_procedure == "IBO":
                    new_orb_mat = lo.ibo.ibo(
                        pyscf_mol, occ_orb_arr, **self.pyscf_calc_params.orb_kwargs
                    )
                else:
                    kernel_kwargs = {}
                    if self.localization_procedure == "Boys":
                        loc_obj = lo.boys.Boys(
                            pyscf_mol, mo_coeff=occ_orb_arr
                        )  # TO-DO: self.pyscf_calc_params.orb_kwargs?
                    else:  # Pipek-Mezey
                        loc_obj = lo.pipek.PipekMezey(pyscf_mol, occ_orb_arr)
                        loc_obj.pop_method = "Mulliken"
                        if self.software != "pySCF":
                            kernel_kwargs = {"mo_coeff": occ_orb_arr}
                    loc_obj.kernel(**kernel_kwargs)
                    new_orb_mat = loc_obj.mo_coeff
                orb_mat.append(new_orb_mat)
        return orb_mat

    def create_mats_savefile(self):
        self.mats_created = True
        if self.mats_savefile is not None:
            # TO-DO Check ways for doing it in a less ugly way.
            saved_data = {
                "mo_coeff": self.mo_coeff,
                "mo_occ": self.mo_occ,
                "mo_energy": self.mo_energy,
                "aos": self.aos,
                "ovlp_mat": self.ovlp_mat,
                "orb_mat": self.orb_mat,
                "atom_ao_ranges": self.atom_ao_ranges,
                "e_tot": self.e_tot,
            }
            if self.calc_type in HF_methods:
                saved_data = {
                    **saved_data,
                    "j_mat": self.j_mat,
                    "k_mat": self.k_mat,
                    "fock_mat": self.fock_mat,
                }
            if self.optimize_geometry:
                saved_data["opt_coords"] = self.opt_coords
            dump2pkl(saved_data, self.mats_savefile)

    def generate_orb_reps(self, rep_params, initial_guess_comp=None):
        """Generates orbital representation.

        rep_params  - object of oml_representations.OML_rep_params class containing parameters of the orbital representation.
        """
        if not self.mats_created:
            self.run_calcs(initial_guess_comp=initial_guess_comp)
        self.orb_reps = []
        for spin in range(self.num_spins()):
            #   Generate the array of orbital representations.
            coupling_matrices = None
            if rep_params.propagator_coup_mat:
                coupling_matrices = gen_propagator_based_coup_mats(
                    rep_params, self.mo_coeff[spin], self.mo_energy[spin], self.ovlp_mat
                )
                coupling_matrices = (self.ovlp_mat, *coupling_matrices)
            if rep_params.ofd_coup_mats:
                coupling_matrices = gen_odf_based_coup_mats(
                    rep_params,
                    self.mo_coeff[spin],
                    self.mo_energy[spin],
                    self.mo_occ[spin],
                    self.ovlp_mat,
                )
            if coupling_matrices is None:
                coupling_matrices = (
                    self.fock_mat[spin],
                    self.j_mat[spin] / orb_occ_prop_coeff(self),
                    self.k_mat[spin] / orb_occ_prop_coeff(self),
                )
            cur_orb_rep_array = generate_orb_rep_array(
                self.orb_mat[spin],
                rep_params,
                self.aos,
                self.atom_ao_ranges,
                self.ovlp_mat,
                np.array(coupling_matrices),
            )
            if rep_params.ofd_coup_mats and rep_params.orb_en_adj:
                for orb_id in range(len(cur_orb_rep_array)):
                    cur_Fock_mat = coupling_matrices[1]
                    cur_orb_rep_array[orb_id].orbital_energy_readjustment(
                        cur_Fock_mat, rep_params
                    )
            self.orb_reps += cur_orb_rep_array
            orb_occ = orb_occ_prop_coeff(self)
            for orb_rep_counter in range(len(self.orb_reps)):
                if not self.orb_reps[orb_rep_counter].virtual:
                    self.orb_reps[orb_rep_counter].rho = orb_occ
            if rep_params.atom_sorted_pseudo_orbs:
                self.orb_reps = gen_atom_sorted_pseudo_orbs(self.orb_reps)

    #   Find maximal value of angular momentum for AOs of current molecule.
    def find_max_angular_momentum(self):
        if not self.mats_created:
            self.run_calcs()
        max_angular_momentum = 0
        for ao in self.aos:
            max_angular_momentum = max(max_angular_momentum, ao.angular)
        return max_angular_momentum

    def generate_pyscf_mol(self):
        # Convert between the Mole class used in pySCF and the Compound class used in the rest of QML.
        mol = gto.Mole()
        # atom_coords should be in Angstrom.
        mol.atom = [
            [atom_type, atom_coords]
            for atom_type, atom_coords in zip(self.atomtypes, self.coordinates)
        ]
        mol.charge = self.charge
        mol.spin = self.spin
        mol.basis = self.basis
        try:
            mol.build()
        except KeyError as KE:
            if str(KE)[1:-1] == mol.basis:
                import basis_set_exchange as bse

                # WARNING: was never used, therefore not %100 sure it works.
                mol.basis = bse.get_basis(mol.basis, fmt="nwchem")
                mol.build()
            else:
                raise KE
        return mol

    def generate_pyscf_mf(self, pyscf_mol, initial_guess_comp=None):
        mf = mf_creator[self.calc_type](pyscf_mol)
        if self.calc_type in KS_methods:
            mf.xc = self.dft_xc
            mf.nlc = self.dft_nlc
        mf.chkfile = self.pyscf_chkfile
        mf.conv_tol = self.pyscf_calc_params.scf_conv_tol
        if self.use_Huckel:
            mf.init_guess = "huckel"
            mf.max_cycle = -1
        else:
            mf.max_cycle = self.pyscf_calc_params.scf_max_cycle
            if ext_isfile(self.pyscf_chkfile):
                mf.init_guess = "chkfile"
        if self.solvent_eps is not None:
            from pyscf.solvent import DDCOSMO

            mf = DDCOSMO(mf)
            mf.with_solvent.eps = self.solvent_eps
        # TO-DO why this does not work???
        if initial_guess_comp is not None:
            dm_init_guess = create_dm_init_guess(
                mf.make_rdm1, initial_guess_comp, self.num_spins()
            )
            mf.kernel(dm_init_guess)
        else:
            mf.run()
        if not (mf.converged or self.use_Huckel):
            mf = scf.newton(mf)
            mf.run()
            if mf.converged:
                print("SCF converged with SO-SCF.")
            else:
                raise pySCFNotConvergedError(pySCFNotConvergedErrorMessage(self))
        #        subprocess.run(["rm", '-f', chkfile])
        return mf

    def generate_pyscf_mf_mol(self, initial_guess_comp=None):
        if self.software == "xTB":
            return xtb_generate_pyscf_mf_mol(self)
        if self.software == "tblite":
            return tblite_generate_pyscf_mf_mol(self)
        if ext_isfile(self.full_pyscf_chkfile):
            return loadpkl(self.full_pyscf_chkfile)
        pyscf_mol = self.generate_pyscf_mol()
        mf = self.generate_pyscf_mf(pyscf_mol, initial_guess_comp=initial_guess_comp)
        if self.optimize_geometry:
            from pyscf.geomopt.geometric_solver import optimize

            pyscf_mol = optimize(mf)
            self.opt_coords = pyscf_mol.atom_coords(unit="Ang")
            mf = self.generate_pyscf_mf(pyscf_mol)
        output = (mf, pyscf_mol)
        if self.full_pyscf_chkfile is not None:
            dump2pkl(output, self.full_pyscf_chkfile)
        return output

    def alter_mo_occ(self, mo_occ):
        if self.calc_type not in unrestricted_methods:
            true_mo_occ = mo_occ
        else:
            true_mo_occ = mo_occ[0]
        for i, orb_occ in enumerate(true_mo_occ):
            if orb_occ < neglect_orb_occ:
                if self.calc_type not in unrestricted_methods:
                    if add_LUMO(self):
                        mo_occ[i] = 2.0
                    if remove_HOMO(self):
                        mo_occ[i - 1] = 0.0
                else:
                    if add_LUMO(self):
                        mo_occ[0][i] = 1.0
                    if remove_HOMO(self):
                        mo_occ[0][i - 1] = 0.0
                break
        return mo_occ

    def occ_orbs(self):
        output = []
        for mo_occ_arr, mo_coeff_arr in zip(self.mo_occ, self.mo_coeff):
            cur_occ_orbs = []
            for basis_func in mo_coeff_arr:
                cur_line = []
                for occ, orb_coeff in zip(mo_occ_arr, basis_func):
                    if occ > neglect_orb_occ:
                        cur_line.append(orb_coeff)
                cur_occ_orbs.append(cur_line)
            output.append(np.array(cur_occ_orbs))
        return output

    def HOMO_en(self):
        # HOMO energy.
        return self.mo_energy[0][self.LUMO_orb_id() - 1]

    def LUMO_en(self):
        # LUMO energy.
        return self.mo_energy[0][self.LUMO_orb_id()]

    def HOMO_LUMO_gap(self):
        return self.LUMO_en() - self.HOMO_en()

    def LUMO_orb_id(self):
        # index of the LUMO orbital.
        for orb_id, occ in enumerate(self.mo_occ[0]):
            if occ < neglect_orb_occ:
                return orb_id

    def adjust_spin_mat_dimen(self, matrices):
        if self.calc_type not in unrestricted_methods:
            return np.array([matrices])
        else:
            return matrices

    def num_spins(self):
        if self.calc_type not in unrestricted_methods:
            return 1
        else:
            return 2


def create_dm_init_guess(rdm_maker, initial_guess_comp, nspins):
    copied_nspins = initial_guess_comp.num_spins()
    dm = []
    for spin_id in range(nspins):
        if copied_nspins == 1:
            copied_spin_id = 0
        else:
            copied_spin_id = spin_id
        dm.append(
            rdm_maker(
                mo_coeff=np.array(initial_guess_comp.mo_coeff[copied_spin_id]),
                mo_occ=np.array(initial_guess_comp.mo_occ[copied_spin_id]),
            )
        )
    if nspins == 1:
        return dm[0]
    else:
        return np.array(dm)


def remove_HOMO(oml_comp):
    used_orb_type = oml_comp.used_orb_type
    return (used_orb_type == "HOMO_removed") or (used_orb_type == "first_excitation")


def add_LUMO(oml_comp):
    used_orb_type = oml_comp.used_orb_type
    return (used_orb_type == "LUMO_added") or (used_orb_type == "first_excitation")


def orb_occ_prop_coeff(comp):
    if comp.calc_type not in unrestricted_methods:
        return 2.0
    else:
        return 1.0


def ext_isfile(filename):
    if filename is None:
        return False
    else:
        return isfile(filename)


class OML_pyscf_calc_params:
    #   Parameters of pySCF calculations.
    def __init__(
        self,
        orb_max_iter=5000,
        orb_grad_tol=1.0e-8,
        scf_max_cycle=5000,
        scf_conv_tol=1e-9,
        scf_conv_tol_grad=None,
    ):
        self.orb_kwargs = {"max_iter": orb_max_iter, "grad_tol": orb_grad_tol}
        self.scf_max_cycle = scf_max_cycle
        self.scf_conv_tol = scf_conv_tol


class OML_Slater_pair:
    def __init__(
        self,
        second_oml_comp_kwargs={},
        initial_guess_from_first=False,
        **first_oml_comp_kwargs
    ):
        self.initial_guess_from_first = initial_guess_from_first
        comp1 = OML_compound(**first_oml_comp_kwargs)

        used_second_oml_comp_kwargs = copy.copy(first_oml_comp_kwargs)
        for second_key in second_oml_comp_kwargs:
            used_second_oml_comp_kwargs[second_key] = second_oml_comp_kwargs[second_key]

        comp2 = OML_compound(**used_second_oml_comp_kwargs)
        self.comps = [comp1, comp2]

    def run_calcs(self):
        self.comps[0].run_calcs()
        if self.initial_guess_from_first:
            initial_guess_comp = self.comps[0]
        else:
            initial_guess_comp = None
        self.comps[1].run_calcs(initial_guess_comp=initial_guess_comp)

    def generate_orb_reps(self, rep_params):
        self.comps[0].generate_orb_reps(rep_params)
        if self.initial_guess_from_first:
            initial_guess_comp = self.comps[0]
        else:
            initial_guess_comp = None
        self.comps[1].generate_orb_reps(
            rep_params, initial_guess_comp=initial_guess_comp
        )


# Additional constructors.


def ASE2OML_compound(ase_in, **other_kwargs):
    return OML_compound(
        coordinates=ase_in.get_positions(),
        atomtypes=ase_in.get_chemical_symbols(),
        **other_kwargs
    )
