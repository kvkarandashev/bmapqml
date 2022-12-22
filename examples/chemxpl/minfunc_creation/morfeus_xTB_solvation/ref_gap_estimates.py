# A script for generating reference gap values used in constraints.
from bmapqml.chemxpl.utils import SMILES_to_egc
from bmapqml.chemxpl.random_walk import TrajectoryPoint
from bmapqml.chemxpl.minimized_functions.morfeus_quantity_estimates import (
    morfeus_FF_xTB_code_quants,
)
import sys
import numpy as np

if len(sys.argv) == 1:
    solvent = None
else:
    solvent = sys.argv[1]

ref_SMILES = ["C=CC=CC=CC=C", "C1=CC=CC=C1"]

# ref_SMILES=["CO"]

gap_name = "HOMO_LUMO_gap"

num_attempts = 512

for SMILES in ref_SMILES:
    ref_tp = TrajectoryPoint(egc=SMILES_to_egc(SMILES))
    ref_data = morfeus_FF_xTB_code_quants(
        ref_tp,
        num_conformers=32,
        num_attempts=num_attempts,
        ff_type="MMFF94",
        quantities=[gap_name],
        remaining_rho=0.9,
        solvent=solvent,
    )
    print(
        SMILES,
        " gap:",
        ref_data["mean"][gap_name],
        "pm",
        ref_data["std"][gap_name] / np.sqrt(num_attempts),
    )
