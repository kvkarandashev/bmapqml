from bmapqml.chemxpl.utils import SMILES_to_egc
from bmapqml.chemxpl.random_walk import TrajectoryPoint
from bmapqml.chemxpl.minimized_functions.morfeus_quantity_estimates import (
    morfeus_FF_xTB_code_quants,
    LinComb_Morfeus_xTB_code,
)
from bmapqml.chemxpl.minimized_functions.toy_problems import NumHAtoms
from bmapqml.chemxpl.minimized_functions.quantity_estimates import ConstrainedQuant
from bmapqml.chemxpl.minimized_functions.mol_constraints import NoProtonation
import numpy as np

SMILES_list = [
    "F",
    "O",
    "N",
    "C",
    "CF",
    "FCF",
    "FC(F)F",
    "FC(F)(F)F",
    "CC",
    "FC(F)C",
    "FCCF",
    "FC(F)C(F)F",
    "FC(F)(F)C(F)(F)F",
]

num_attempts = 4

num_conformers = 50

min_func_quantities = ["solvation_energy", "HOMO_LUMO_gap"]

min_func_coefficients = [1.0, -1.0]

min_func_add_mults = [NumHAtoms(), None]

min_func_add_mult_powers = [-1, 1]

all_vals = {"normalized_solvation_energy": []}
for quant in min_func_quantities:
    all_vals[quant] = []

print("PROPERTY CALCULATIONS")
for solvent in ["water", "ether", "dmso", None]:
    print("SOLVENT", solvent, ":")
    cur_quantities = ["energy", "HOMO_LUMO_gap", "solvation_energy", "dipole"]
    for SMILES in SMILES_list:
        tp = TrajectoryPoint(egc=SMILES_to_egc(SMILES))
        res_dict = morfeus_FF_xTB_code_quants(
            tp,
            num_conformers=num_conformers,
            num_attempts=num_attempts,
            quantities=cur_quantities,
            solvent=solvent,
        )
        print("SMILES", SMILES, ":")
        for quant in cur_quantities:
            print(quant, ":", res_dict["mean"][quant], res_dict["std"][quant])
            if quant in min_func_quantities:
                all_vals[quant].append(res_dict["mean"][quant])
        all_vals["normalized_solvation_energy"].append(
            all_vals["solvation_energy"][-1] / tp.egc.num_heavy_atoms()
        )

for quant_id, quant in enumerate(min_func_quantities):
    min_func_coefficients[quant_id] /= np.std(all_vals[quant])

print("MINIMIZED FUNCTION CALCULATIONS")

for solvent in ["water", "ether", "dmso"]:
    print("SOLVENT:", solvent)
    for solvation_energy_normalized in [True, False]:
        print("SOLVATION ENERGY NORMALIZED:", solvation_energy_normalized)
        if solvation_energy_normalized:
            min_func_add_mults = [NumHAtoms(), None]
            min_func_add_mult_powers = [-1, 1]
            used_quants = ["normalized_solvation_energy", "HOMO_LUMO_gap"]
        else:
            min_func_add_mults = None
            min_func_add_mult_powers = None
            used_quants = ["solvation_energy", "HOMO_LUMO_gap"]

        min_func_coefficients = [1.0, -1.0]

        for quant_id, quant in enumerate(used_quants):
            min_func_coefficients[quant_id] /= np.std(all_vals[quant])

        min_func_true = LinComb_Morfeus_xTB_code(
            quantities=min_func_quantities,
            solvent=solvent,
            coefficients=min_func_coefficients,
            num_attempts=num_attempts,
            num_conformers=num_conformers,
            add_mult_funcs=min_func_add_mults,
            add_mult_func_powers=min_func_add_mult_powers,
        )

        min_func = ConstrainedQuant(
            min_func_true, "morfeus_lin_comb", [NoProtonation([8, 9])]
        )

        for SMILES in SMILES_list:
            tp = TrajectoryPoint(egc=SMILES_to_egc(SMILES))
            print(SMILES, min_func(tp))  # , tp.calculated_data["xTB_res"])
