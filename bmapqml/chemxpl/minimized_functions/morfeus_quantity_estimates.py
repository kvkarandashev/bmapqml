from morfeus.conformer import conformers_from_rdkit
from ..utils import (
    chemgraph_to_canonical_rdkit,
    InvalidAdjMat,
    chemgraph_from_ncharges_coords,
)
from ...utils import NUCLEAR_CHARGE, checked_environ_val
from .xtb_quantity_estimates import FF_xTB_HOMO_LUMO_gap, FF_xTB_dipole
import numpy as np


def morfeus_coord_info_from_tp(
    tp,
    num_attempts=1,
    ff_type="MMFF94",
    return_rdkit_obj=False,
    all_confs=False,
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
        conformers = conformers_from_rdkit(
            canon_rdkit_mol,
            n_confs=num_attempts,
            optimize=ff_type,
            n_threads=checked_environ_val("OMP_NUM_THREADS", default_answer=1),
        )
    except Exception as ex:
        if not isinstance(ex, ValueError):
            print("#PROBLEMATIC_MORFEUS:", tp)
        return output

    nuclear_charges = np.array([NUCLEAR_CHARGE[el] for el in conformers[0]])
    output["nuclear_charges"] = nuclear_charges

    min_en_id = np.argmin(conformers[2])
    coordinates = conformers[1][min_en_id]

    try:
        coord_based_cg = chemgraph_from_ncharges_coords(nuclear_charges, coordinates)
    except InvalidAdjMat:
        return output
    if coord_based_cg != cg:
        return output

    if all_confs:
        output["coordinates"] = conformers[1]
        output["rdkit_energy"] = conformers[2]
    else:
        output["coordinates"] = coordinates
        output["rdkit_energy"] = conformers[2][min_en_id]

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
from xtb.interface import Calculator
from xtb.utils import get_method, get_solvent
from ...utils import any_element_in_list
from ...data import conversion_coefficient


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

    return calc_obj.singlepoint()


def xTB_quants(
    coordinates,
    nuclear_charges,
    quantities=[],
    solvent=None,
    **other_xTB_singlepoint_res
):
    res = xTB_singlepoint_res(
        coordinates, nuclear_charges, **other_xTB_singlepoint_res, solvent=solvent
    )

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
        en_nosolvent = res_nosolvent.get_energy()
        output["energy_no_solvent"] = en_nosolvent
        output["solvation_energy"] = output["energy"] - en_nosolvent

    return output


def morfeus_FF_xTB_code_quants(
    tp,
    num_conformers=1,
    num_attempts=1,
    ff_type="MMFF94",
    quantities=[],
    **xTB_quants_args
):
    """
    Use morfeus-ml FF coordinates with Grimme lab's xTB code to calculate some quantities.
    """
    quant_arrs = {}
    for quant in quantities:
        quant_arrs[quant] = np.empty((num_attempts,))

    for i in range(num_attempts):
        coord_info = morfeus_coord_info_from_tp(
            tp, num_attempts=num_conformers, ff_type=ff_type
        )
        coordinates = coord_info["coordinates"]
        if coordinates is None:
            for quant in quantities:
                quant_arrs[quant] = None
            break
        res = xTB_quants(
            coordinates,
            coord_info["nuclear_charges"],
            quantities=quantities,
            **xTB_quants_args
        )
        for quant in quantities:
            quant_arrs[quant][i] = res[quant]

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
        add_mult_funcs=None,
        add_mult_func_powers=None,
        xTB_res_dict_name="xTB_res",
        **xTB_related_kwargs
    ):
        self.quantities = quantities
        self.coefficients = coefficients
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
        xTB_res_dict = trajectory_point_in.calc_or_lookup(
            {self.xTB_res_dict_name: morfeus_FF_xTB_code_quants},
            kwargs_dict={
                self.xTB_res_dict_name: {
                    **self.xTB_related_kwargs,
                    "quantities": self.quantities,
                }
            },
        )[self.xTB_res_dict_name]
        result = 0.0
        self.call_counter += 1
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
