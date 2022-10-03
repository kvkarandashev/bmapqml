from bmapqml.utils import loadpkl, dump2pkl
from bmapqml.chemxpl.minimized_functions.quantity_estimates import ConstrainedQuant
from bmapqml.chemxpl.minimized_functions.mol_constraints import NoProtonation
from bmapqml.chemxpl.minimized_functions.morfeus_quantity_estimates import (
    LinComb_Morfeus_xTB_code,
)
import numpy as np
import sys

data_file_name = sys.argv[1]

solvent = sys.argv[2]

all_data = loadpkl(data_file_name)

quantities_of_interest = ["HOMO_LUMO_gap", "solvation_energy"]

signs = {"HOMO_LUMO_gap": -1, "solvation_energy": 1}

stddevs = {}

for quant in quantities_of_interest:
    all_vals = []
    for entry in all_data:
        if "mean" in entry:
            cur_val = entry["mean"][quant]
            if cur_val is not None:
                all_vals.append(entry["mean"][quant])
    stddevs[quant] = np.std(all_vals)

func_component_lists = {
    "electrolyte": quantities_of_interest,
    "gap": ["HOMO_LUMO_gap"],
    "solvation": ["solvation_energy"],
}

constraints = [NoProtonation([8, 9])]

for func_name, func_component_list in func_component_lists.items():
    coeffs = []
    for quant in func_component_list:
        cur_sign = signs[quant] / stddevs[quant]
        coeffs.append(cur_sign)
    lin_comb = LinComb_Morfeus_xTB_code(
        func_component_list,
        coeffs,
        num_attempts=1,
        num_conformers=8,
        ff_type="MMFF94",
        solvent=solvent,
    )
    constr_quant = ConstrainedQuant(lin_comb, func_name, constraints=constraints)
    dump2pkl(constr_quant, "constr_" + func_name + "_" + data_file_name)
