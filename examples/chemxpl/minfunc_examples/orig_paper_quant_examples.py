# For a given SMILES
import sys
from bmapqml.chemxpl.rdkit_utils import SMILES_to_egc
from bmapqml.chemxpl.random_walk import TrajectoryPoint
from bmapqml.chemxpl.minimized_functions.morfeus_quantity_estimates import (
    morfeus_FF_xTB_code_quants,
)

gap_constraints = {"weak": 0.08895587351640835, "strong": 0.17734766509696615}

SMILES = sys.argv[1]

est_quant = sys.argv[2]
if est_quant not in ["dipole", "solvation_energy"]:
    print("Invalid estimated quantity")
    quit()
quantities = [
    est_quant,
    "HOMO_LUMO_gap",
]
# More available quantites: "energy", "energy_no_solvent", "atomization_energy", "normalized_atomization_energy", "num_evals"


# These are parameters corresponding to the overconverged estimates.
kwargs = {
    "ff_type": "MMFF94",
    "remaining_rho": 0.9,
    "num_conformers": 32,
    "num_attempts": 16,
    "solvent": "water",
    "quantities": quantities,
}


egc = SMILES_to_egc(SMILES)

tp = TrajectoryPoint(egc=egc)

results = morfeus_FF_xTB_code_quants(tp, **kwargs)

std_RMSE_coeff = 0.25

for quant in quantities:
    val = results["mean"][quant]
    if quant == "HOMO_LUMO_gap":
        HOMO_LUMO_gap = val
    std = results["std"][quant]
    print(quant, val, "pm", std * std_RMSE_coeff)

for gap_constraint, val in gap_constraints.items():
    print(gap_constraint, "gap constraint satisfied:", HOMO_LUMO_gap > val)
