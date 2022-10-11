from bmapqml.chemxpl.utils import SMILES_to_egc, xyz2mol_extgraph
from bmapqml.chemxpl.random_walk import TrajectoryPoint
from bmapqml.chemxpl.minimized_functions.morfeus_quantity_estimates import (
    morfeus_FF_xTB_code_quants,
)
import numpy as np
import sys

flag = "None"

if sys.argv[1] in ["--SMILES", "--xyz"]:
    flag_num = 1
    flag = sys.argv[1]
else:
    flag_num = 0

if flag == "--xyz":
    xyz = sys.argv[1 + flag_num]
    egc = xyz2mol_extgraph(xyz)
else:
    SMILES = sys.argv[1 + flag_num]
    egc = SMILES_to_egc(SMILES)

if len(sys.argv) > 2 + flag_num:
    solvent = sys.argv[2 + flag_num]
else:
    solvent = "water"

num_attempts = 4

num_conformers = 50

quantities = [
    "energy",
    "HOMO_LUMO_gap",
    "solvation_energy",
    "dipole",
    "atomization_energy",
    "normalized_atomization_energy",
]
tp = TrajectoryPoint(egc=egc)
res_dict = morfeus_FF_xTB_code_quants(
    tp,
    num_conformers=num_conformers,
    num_attempts=num_attempts,
    remaining_rho=0.95,
    quantities=quantities,
    solvent=solvent,
)
for quant in quantities:
    print(quant, ":", res_dict["mean"][quant], res_dict["std"][quant])
