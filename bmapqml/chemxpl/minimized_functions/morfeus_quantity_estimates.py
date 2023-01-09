# TODO rewrite all functions in terms of classes that track number of calls and successful coordinate construction?

from morfeus.conformer import ConformerEnsemble, K_B, HARTREE
from ..utils import (
    chemgraph_to_canonical_rdkit,
    InvalidAdjMat,
    chemgraph_from_ncharges_coords,
)
from ...utils import (
    checked_environ_val,
    repeated_dict,
    all_None_dict,
    any_element_in_list,
)
from .xtb_quantity_estimates import FF_xTB_HOMO_LUMO_gap, FF_xTB_dipole
import numpy as np
import copy, os
from ...data import room_T

#   The module was originally written in a form that does not produce any numpy warning for testing purposes. Kept like this for reproducability's sake.
np.seterr(all="raise")


def exp_wexceptions(val):
    """
    A version of numpy.exp that does not raise FloatingPointError exceptions.
    """
    try:
        return np.exp(val)
    except FloatingPointError:
        if val > 0.0:
            return np.inf
        else:
            return 0.0


def renorm_wexceptions(array):
    """
    Normalize a numpy array; exceptions are accounted for.
    """
    s = np.sum(array)
    for i in range(array.shape[0]):
        try:
            array[i] /= s
        except FloatingPointError:
            array[i] = 0.0


def morfeus_coord_info_from_tp(
    tp,
    num_attempts=1,
    ff_type="MMFF94",
    return_rdkit_obj=False,
    all_confs=False,
    temperature=room_T,
    coord_validity_check=True,
    **dummy_kwargs
):
    """
    Coordinates corresponding to a TrajectoryPoint object
    tp : TrajectoryPoint object
    num_attempts : number of attempts taken to generate MMFF coordinates (introduced because for QM9 there is a ~10% probability that the coordinate generator won't converge)
    **kwargs : keyword arguments for the egc_with_coords procedure
    """
    output = {"coordinates": None, "nuclear_charges": None, "canon_rdkit_SMILES": None}
    cg = tp.egc.chemgraph
    canon_rdkit_mol, _, _, canon_rdkit_SMILES = chemgraph_to_canonical_rdkit(cg)
    output["canon_rdkit_SMILES"] = canon_rdkit_SMILES
    if return_rdkit_obj:
        output["canon_rdkit_mol"] = canon_rdkit_mol
    # TODO: better place to check OMP_NUM_THREADS value?
    try:
        conformers = ConformerEnsemble.from_rdkit(
            canon_rdkit_mol,
            n_confs=num_attempts,
            optimize=ff_type,
            n_threads=checked_environ_val("MORFEUS_NUM_THREADS", default_answer=1),
        )
    except Exception as ex:
        if not isinstance(ex, ValueError):
            print("#PROBLEMATIC_MORFEUS:", tp)
        return output
    if all_confs:
        conformers.prune_rmsd()
    all_coordinates = conformers.get_coordinates()
    energies = conformers.get_energies()

    nuclear_charges = np.array(conformers.elements)
    output["nuclear_charges"] = nuclear_charges

    if len(energies) == 0:
        return output

    min_en_id = np.argmin(energies)

    min_en = energies[min_en_id]
    min_coordinates = all_coordinates[min_en_id]

    if coord_validity_check:
        try:
            coord_based_cg = chemgraph_from_ncharges_coords(
                nuclear_charges, min_coordinates
            )
        except InvalidAdjMat:
            return output
        if coord_based_cg != cg:
            return output

    if all_confs:
        output["coordinates"] = all_coordinates
        output["rdkit_energy"] = energies
        output["rdkit_degeneracy"] = conformers.get_degeneracies()
        #        output["rdkit_Boltzmann"] = conformers.boltzmann_weights(
        #            temperature=temperature
        #        )
        # Rewriting in terms of exp_wexceptions because errors occur otherwise.
        boltzmann_factors = np.array(
            [
                exp_wexceptions(-(en - min_en) / temperature / K_B * HARTREE)
                for en in energies
            ]
        )
        renorm_wexceptions(boltzmann_factors)
        output["rdkit_Boltzmann"] = boltzmann_factors
    else:
        output["coordinates"] = all_coordinates[min_en_id]
        output["rdkit_energy"] = min_en
    return output


xTB_quant_morfeus_kwargs = {
    "coord_calculation_type": "morfeus",
    "coord_info_from_tp_func": morfeus_coord_info_from_tp,
}


def morfeus_FF_xTB_HOMO_LUMO_gap(**kwargs):
    return FF_xTB_HOMO_LUMO_gap(**xTB_quant_morfeus_kwargs, **kwargs)


def morfeus_FF_xTB_dipole(**kwargs):
    return FF_xTB_dipole(**xTB_quant_morfeus_kwargs, **kwargs)


# from ...interfaces.xtb_interface import xTB_results
from xtb.libxtb import VERBOSITY_MUTED, VERBOSITY_MINIMAL, VERBOSITY_FULL
from xtb.interface import Calculator, XTBException
from xtb.utils import get_method, get_solvent
from ...data import conversion_coefficient
import os
from ...utils import loadpkl, dump2pkl, weighted_array
from ..periodic import supported_ncharges

atom_energy_filename = os.path.dirname(__file__) + "/atom_energies.pkl"

if os.path.isfile(atom_energy_filename):
    atom_energies = loadpkl(atom_energy_filename)
else:
    atom_energies = {}

# TODO check that an atom's spin is handled properly.
def xTB_atom_energy(ncharge, parametrization="gfn2-xtb", solvent=None):
    res = xTB_singlepoint_res(
        np.array([[0.0, 0.0, 0.0]]),
        np.array([ncharge]),
        parametrization=parametrization,
        solvent=solvent,
    )
    return res.get_energy()


def gen_atom_energy(ncharge, parametrization="gfn2-xtb", solvent=None):
    check_tuple = (ncharge, parametrization, solvent)
    if check_tuple not in atom_energies:
        atom_energies[check_tuple] = xTB_atom_energy(
            ncharge, parametrization=parametrization, solvent=solvent
        )
        dump2pkl(atom_energies, atom_energy_filename)


def gen_atom_energies(
    ncharges=supported_ncharges(), parametrization="gfn2-xtb", solvent=None
):
    for ncharge in ncharges:
        gen_atom_energy(ncharge, parametrization=parametrization, solvent=solvent)


def atom_energy(ncharge, parametrization="gfn2-xtb", solvent=None, **dummy_kwargs):
    check_tuple = (ncharge, parametrization, solvent)
    if check_tuple in atom_energies:
        return atom_energies[check_tuple]
    else:
        raise Exception(
            "Atom energy unavailable:",
            check_tuple,
            "; try running gen_atom_energy to add it to database.",
        )


verbosity_dict = {
    "MUTED": VERBOSITY_MUTED,
    "MINIMAL": VERBOSITY_MINIMAL,
    "FULL": VERBOSITY_FULL,
}


def xTB_singlepoint_res(
    coordinates,
    nuclear_charges,
    accuracy=None,
    verbosity="MUTED",
    parametrization="gfn2-xtb",
    solvent=None,
    max_iterations=None,
    electronic_temperature=None,
):
    calc_obj = Calculator(
        get_method(parametrization),
        nuclear_charges,
        coordinates * conversion_coefficient["Angstrom_Bohr"],
    )
    if accuracy is not None:
        calc_obj.set_accuracy(accuracy)
    if max_iterations is not None:
        calc_obj.set_max_iterations(max_iterations)
    if electronic_temperature is not None:
        calc_obj.set_electronic_temperature(electronic_temperature)
    if solvent is not None:
        calc_obj.set_solvent(get_solvent(solvent))
    calc_obj.set_verbosity(verbosity_dict[verbosity])

    try:
        return calc_obj.singlepoint()
    except:
        return None


def xTB_quants(
    coordinates,
    nuclear_charges,
    quantities=[],
    solvent=None,
    **other_xTB_singlepoint_res
):
    res = xTB_singlepoint_res(
        coordinates, nuclear_charges, solvent=solvent, **other_xTB_singlepoint_res
    )

    if res is None:
        return None

    output = {}
    output["energy"] = res.get_energy()

    if any_element_in_list(quantities, "HOMO_energy", "LUMO_energy", "HOMO_LUMO_gap"):
        orbital_energies = res.get_orbital_eigenvalues()
        orbital_occupations = res.get_orbital_occupations()
        HOMO_energy = max(orbital_energies[np.where(orbital_occupations > 0.1)])
        LUMO_energy = min(orbital_energies[np.where(orbital_occupations < 0.1)])
        output["HOMO_energy"] = HOMO_energy
        output["LUMO_energy"] = LUMO_energy
        output["HOMO_LUMO_gap"] = LUMO_energy - HOMO_energy

    if "dipole" in quantities:
        dipole_vec = res.get_dipole()
        output["dipole"] = np.sqrt(np.sum(dipole_vec**2))

    if any_element_in_list(quantities, "solvation_energy", "energy_no_solvent"):
        res_nosolvent = xTB_singlepoint_res(
            coordinates, nuclear_charges, **other_xTB_singlepoint_res, solvent=None
        )
        if res_nosolvent is None:
            return None
        en_nosolvent = res_nosolvent.get_energy()
        output["energy_no_solvent"] = en_nosolvent
        output["solvation_energy"] = output["energy"] - en_nosolvent
    if any_element_in_list(
        quantities, "atomization_energy", "normalized_atomization_energy"
    ):
        atom_en_sum = sum(
            atom_energy(nc, solvent=solvent, **other_xTB_singlepoint_res)
            for nc in nuclear_charges
        )
        output["atomization_energy"] = output["energy"] - atom_en_sum

    return output


class weighted_index:
    def __init__(self, entry_id, w):
        self.entry_id = entry_id
        self.rho = w


def cut_weights(bweights, remaining_rho=None):
    weights = [None for _ in range(len(bweights))]
    if remaining_rho is None:
        min_en_id = np.argmax(bweights)
        weights[min_en_id] = 1.0
    else:
        bweights_list = weighted_array(
            [weighted_index(w_id, w) for w_id, w in enumerate(bweights)]
        )
        bweights_list.normalize_sort_rhos()
        bweights_list.cutoff_minor_weights(remaining_rho=remaining_rho)
        for wi in bweights_list:
            weights[wi.entry_id] = wi.rho
    return weights


num_eval_name = "num_evals"


def morfeus_FF_xTB_code_quants_weighted(
    tp,
    num_conformers=1,
    ff_type="MMFF94",
    quantities=[],
    remaining_rho=None,
    coord_validity_check=True,
    **xTB_quants_kwargs
):
    coord_info = morfeus_coord_info_from_tp(
        tp,
        num_attempts=num_conformers,
        ff_type=ff_type,
        all_confs=True,
        coord_validity_check=coord_validity_check,
    )
    coordinates = coord_info["coordinates"]
    if coordinates is None:
        return None
    conf_weights = cut_weights(
        coord_info["rdkit_Boltzmann"], remaining_rho=remaining_rho
    )
    ncharges = coord_info["nuclear_charges"]
    output = repeated_dict(quantities, 0.0)
    for conf_coords, weight in zip(coord_info["coordinates"], conf_weights):
        if weight is not None:
            #            try:
            res = xTB_quants(
                conf_coords, ncharges, quantities=quantities, **xTB_quants_kwargs
            )
            #            except XTBException:
            #                print("###PROBLEMATIC_MOLECULE:", coord_info["canon_rdkit_SMILES"])
            #                return None
            if res is None:
                print(
                    "###PROBLEMATIC MOLECULE for xTB", coord_info["canon_rdkit_SMILES"]
                )
                return None
            if "normalized_atomization_energy" in quantities:
                res["normalized_atomization_energy"] = (
                    res["atomization_energy"] / tp.egc.chemgraph.tot_ncovpairs()
                )
            for quant in quantities:
                if quant == num_eval_name:
                    output[quant] += 1.0
                else:
                    output[quant] += weight * res[quant]
    return output


def morfeus_FF_xTB_code_quants(
    tp,
    num_conformers=1,
    num_attempts=1,
    ff_type="MMFF94",
    quantities=[],
    remaining_rho=None,
    **xTB_quants_kwargs
):
    """
    Use morfeus-ml FF coordinates with Grimme lab's xTB code to calculate some quantities.
    """
    quant_arrs = repeated_dict(quantities, np.zeros((num_attempts,)), copy_needed=True)

    for i in range(num_attempts):
        av_res = morfeus_FF_xTB_code_quants_weighted(
            tp,
            num_conformers=num_conformers,
            ff_type=ff_type,
            quantities=quantities,
            remaining_rho=remaining_rho,
            **xTB_quants_kwargs
        )
        if av_res is None:
            quant_arrs = all_None_dict(quantities)
            break
        for quant in quantities:
            quant_arrs[quant][i] = av_res[quant]

    output = {"arrs": quant_arrs}
    output["mean"] = {}
    output["std"] = {}
    for quant, quant_arr in quant_arrs.items():
        if quant_arr is None:
            output["mean"][quant] = None
            output["std"][quant] = None
        else:
            try:
                mean = np.mean(quant_arr)
            except FloatingPointError:
                mean = 0.0
            output["mean"][quant] = mean
            try:
                stddev = np.std(quant_arr)
            except FloatingPointError:
                stddev = 0.0
            output["std"][quant] = stddev
    return output


class LinComb_Morfeus_xTB_code:
    """
    Calculate linear combination of several quantities obtained from xTB code with morfeus-ml coordinates.
    """

    def __init__(
        self,
        quantities=[],
        coefficients=[],
        constr_quants=[],
        cq_upper_bounds=None,
        cq_lower_bounds=None,
        add_mult_funcs=None,
        add_mult_func_powers=None,
        xTB_res_dict_name="xTB_res",
        **xTB_related_kwargs
    ):
        self.quantities = quantities
        self.coefficients = coefficients

        self.constr_quants = constr_quants
        self.cq_upper_bounds = cq_upper_bounds
        self.cq_lower_bounds = cq_lower_bounds

        self.needed_quantities = copy.copy(self.quantities)
        for quant in constr_quants:
            if quant not in self.needed_quantities:
                self.needed_quantities.append(quant)

        if add_mult_funcs is None:
            self.add_mult_funcs = [None for _ in range(len(quantities))]
        else:
            self.add_mult_funcs = add_mult_funcs
        if add_mult_func_powers is None:
            self.add_mult_func_powers = [1 for _ in range(len(quantities))]
        else:
            self.add_mult_func_powers = add_mult_func_powers
        self.xTB_res_dict_name = xTB_res_dict_name
        self.xTB_related_kwargs = xTB_related_kwargs
        self.call_counter = 0

    def __call__(self, trajectory_point_in):
        self.call_counter += 1

        xTB_res_dict = trajectory_point_in.calc_or_lookup(
            {self.xTB_res_dict_name: morfeus_FF_xTB_code_quants},
            kwargs_dict={
                self.xTB_res_dict_name: {
                    **self.xTB_related_kwargs,
                    "quantities": self.needed_quantities,
                }
            },
        )[self.xTB_res_dict_name]
        result = 0.0
        for quant_id, quant in enumerate(self.constr_quants):
            cur_val = xTB_res_dict["mean"][quant]
            if cur_val is None:
                return None
            if self.cq_upper_bounds is not None:
                if (self.cq_upper_bounds[quant_id] is not None) and (
                    cur_val > self.cq_upper_bounds[quant_id]
                ):
                    return None
            if self.cq_lower_bounds is not None:
                if (self.cq_lower_bounds[quant_id] is not None) and (
                    cur_val < self.cq_lower_bounds[quant_id]
                ):
                    return None
        for quant, coeff, add_mult, add_mult_power in zip(
            self.quantities,
            self.coefficients,
            self.add_mult_funcs,
            self.add_mult_func_powers,
        ):
            cur_val = xTB_res_dict["mean"][quant]
            if cur_val is None:
                return None
            cur_add = cur_val * coeff
            if add_mult is not None:
                cur_add *= add_mult(trajectory_point_in) ** add_mult_power
            result += cur_add
        return result

    def __str__(self):
        output = (
            "LinComb_Morfeus_xTB_code\nQuants: "
            + str(self.quantities)
            + "\nCoefficients: "
            + str(self.coefficients)
            + "\n"
        )
        if self.constr_quants:
            output += "Constraints:\n"
            for quant_id, quant_name in enumerate(self.constr_quants):
                output += quant_name + " "
                if self.cq_lower_bounds is not None:
                    output += str(self.cq_lower_bounds[quant_id])
                output += ":"
                if self.cq_upper_bounds is not None:
                    output += str(self.cq_upper_bounds[quant_id])
                output += "\n"
        output += "Calculation parameters:\n"
        for k, v in self.xTB_related_kwargs.items():
            output += k + " " + str(v) + "\n"
        return output

    def __repr__(self):
        return str(self)


from ..random_walk import TrajectoryPoint, ChemGraph


class Hydrogenation:
    def __init__(
        self,
        num_attempts=1,
        num_conformers=1,
        remaining_rho=None,
        energy_av_std_key="energy_av_std",
        **morfeus_related_kwargs
    ):
        """
        Returns energy of hydrogenation via MMFF.
        """
        self.morfeus_related_kwargs = morfeus_related_kwargs
        self.num_attempts = num_attempts
        self.num_conformers = num_conformers
        self.remaining_rho = remaining_rho
        self.energy_av_std_key = energy_av_std_key
        self.call_counter = 0

        from ..valence_treatment import h2chemgraph

        self.h2chemgraph = h2chemgraph()

    def h2energy(self):
        return self.average_energy(
            TrajectoryPoint(cg=self.h2chemgraph), coord_validity_check=False
        )

    def hydrogenated_species_energy(self, cg):
        return self.average_energy(TrajectoryPoint(cg=cg))

    def energy_vals(self, tp_in, coord_validity_check=True):
        energy_arr = np.zeros((self.num_attempts,))
        for i in range(self.num_attempts):
            morfeus_data = morfeus_coord_info_from_tp(
                tp_in,
                num_attempts=self.num_conformers,
                all_confs=True,
                coord_validity_check=coord_validity_check,
                **self.morfeus_related_kwargs
            )
            coords = morfeus_data["coordinates"]
            if coords is None:
                return None, None
            conf_weights = cut_weights(
                morfeus_data["rdkit_Boltzmann"], remaining_rho=self.remaining_rho
            )
            for en, weight in zip(morfeus_data["rdkit_energy"], conf_weights):
                if weight is not None:
                    energy_arr[i] += en * weight
        return np.mean(energy_arr), np.std(energy_arr)

    def average_energy(self, tp_in, coord_validity_check=True):
        av, _ = tp_in.calc_or_lookup(
            {self.energy_av_std_key: self.energy_vals},
            kwargs_dict={
                self.energy_av_std_key: {"coord_validity_check": coord_validity_check}
            },
        )[self.energy_av_std_key]
        return av

    def __call__(self, tp_in):
        self.call_counter += 1
        cur_ncharges = {}
        for ha in tp_in.chemgraph().hatoms:
            ncharge = ha.ncharge
            if ncharge not in cur_ncharges:
                cur_ncharges[ncharge] = 0
            cur_ncharges[ncharge] += 1
        output = self.average_energy(tp_in)
        if output is None:
            return None
        extra_nhydrogens = -tp_in.chemgraph().tot_nhydrogens()
        # Note that we do not pre-initialize energies of individual hydrogenated species to allow for statistical error cancelation.
        for ncharge, natoms in cur_ncharges.items():
            temp_cg = ChemGraph(
                nuclear_charges=np.array([ncharge]),
                hydrogen_autofill=True,
                adj_mat=np.array([[0]]),
            )
            output -= self.hydrogenated_species_energy(temp_cg) * natoms
            extra_nhydrogens += temp_cg.tot_nhydrogens() * natoms
        if extra_nhydrogens % 2 != 0:
            raise Exception()
        output += self.h2energy() * extra_nhydrogens / 2
        return output


class Hydrogenation_xTB(Hydrogenation):
    def __init__(
        self,
        xTB_related_kwargs={},
        xTB_related_overconverged_kwargs={},
        xTB_res_dict_name="xTB_res",
    ):
        """
        Use forcefield coordinates and xTB to estimate hydrogenation energy.
        """
        self.xTB_related_kwargs = xTB_related_kwargs
        self.xTB_related_overconverged_kwargs = xTB_related_overconverged_kwargs
        self.call_counter = 0
        self.xTB_res_dict_name = xTB_res_dict_name

        self.hydrogenated_energies_precalc = {}

        from ..valence_treatment import h2chemgraph

        self.h2_energy_precalc = self.overconverged_average_energy(h2chemgraph())

    def average_energy(self, tp_in, xTB_kwargs=None):
        if xTB_kwargs is None:
            xTB_kwargs = self.xTB_related_kwargs
        xTB_res_dict = tp_in.calc_or_lookup(
            {self.xTB_res_dict_name: morfeus_FF_xTB_code_quants},
            kwargs_dict={
                self.xTB_res_dict_name: {
                    **xTB_kwargs,
                    "quantities": ["energy"],
                }
            },
        )[self.xTB_res_dict_name]
        return xTB_res_dict["mean"]["energy"]

    def overconverged_average_energy(self, cg):
        xTB_kwargs = {
            "coord_validity_check": False,
            **self.xTB_related_overconverged_kwargs,
        }
        temp_tp = TrajectoryPoint(cg=cg)
        return self.average_energy(temp_tp, xTB_kwargs=xTB_kwargs)

    def h2energy(self):
        return self.h2_energy_precalc

    def hydrogenated_species_energy(self, cg):
        core_ncharge = cg.hatoms[0].ncharge
        if core_ncharge not in self.hydrogenated_energies_precalc:
            self.hydrogenated_energies_precalc[
                core_ncharge
            ] = self.overconverged_average_energy(cg)
        return self.hydrogenated_energies_precalc[core_ncharge]
