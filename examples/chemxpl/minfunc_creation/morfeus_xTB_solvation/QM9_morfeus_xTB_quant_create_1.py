from bmapqml.utils import loadpkl, dump2pkl
from bmapqml.chemxpl.minimized_functions.quantity_estimates import ConstrainedQuant
from bmapqml.chemxpl.minimized_functions.mol_constraints import (
    NoProtonation,
    NoForbiddenBonds,
)
from bmapqml.chemxpl.minimized_functions.morfeus_quantity_estimates import (
    LinComb_Morfeus_xTB_code,
    morfeus_FF_xTB_code_quants,
)
from bmapqml.chemxpl.valence_treatment import str2ChemGraph
from bmapqml.chemxpl.utils import SMILES_to_egc
from bmapqml.chemxpl.random_walk import TrajectoryPoint
import numpy as np
import sys

data_file_name = sys.argv[1]

solvent = sys.argv[2]

all_data = loadpkl(data_file_name)

quantities_of_interest = ["dipole", "solvation_energy"]

gap_constraints = ["weak", "strong"]

reference_gap_molecules = {"weak": "C=CC=CC=CC=C", "strong": "C1=CC=CC=C1"}

gap_name = "HOMO_LUMO_gap"

no_prot = [8, 9]
forbidden_bonds = [(7, 7), (8, 8), (9, 9), (7, 8), (7, 9), (8, 9)]

signs = {"dipole": -1, "solvation_energy": 1}

stddevs = {}

constraints = [
    NoProtonation(restricted_ncharges=no_prot),
    NoForbiddenBonds(forbidden_bonds=forbidden_bonds),
]


def constraints_broken(tp):
    for constr in constraints:
        if not constr(tp):
            return True
    return False


for quant in quantities_of_interest:
    all_vals = []
    for entry in all_data:
        if "mean" in entry:
            cur_chemgraph = str2ChemGraph(entry["chemgraph"])
            tp = TrajectoryPoint(cg=cur_chemgraph)
            if constraints_broken(tp):
                continue
            cur_val = entry["mean"][quant]
            if cur_val is not None:
                all_vals.append(entry["mean"][quant])
    stddevs[quant] = np.std(all_vals)

func_component_lists = {
    "dipsolv": quantities_of_interest,
    "dipole": ["dipole"],
    "solvation": ["solvation_energy"],
}

for gap_constraint in gap_constraints:
    ref_SMILES = reference_gap_molecules[gap_constraint]
    ref_tp = TrajectoryPoint(egc=SMILES_to_egc(ref_SMILES))
    ref_gap = morfeus_FF_xTB_code_quants(
        ref_tp,
        num_conformers=16,
        num_attempts=1,
        ff_type="MMFF94",
        quantities=[gap_name],
        remaining_rho=0.9,
    )["mean"][gap_name]
    print("Constraining gaps:", gap_constraint, ref_gap, ref_SMILES)

    for func_name, func_component_list in func_component_lists.items():
        coeffs = []
        for quant in func_component_list:
            cur_sign = signs[quant] / stddevs[quant]
            coeffs.append(cur_sign)
        lin_comb = LinComb_Morfeus_xTB_code(
            func_component_list,
            coeffs,
            constr_quants=[gap_name],
            cq_lower_bounds=[ref_gap],
            num_attempts=1,
            num_conformers=8,
            ff_type="MMFF94",
            solvent=solvent,
        )
        constr_quant = ConstrainedQuant(lin_comb, func_name, constraints=constraints)
        dump2pkl(
            constr_quant,
            "constr_"
            + func_name
            + "_gap_constr_"
            + str(gap_constraint)
            + "_"
            + data_file_name,
        )
