from bmapqml.chemxpl.utils import SMILES_to_egc
from bmapqml.chemxpl.random_walk import TrajectoryPoint
from bmapqml.chemxpl.minimized_functions.rmgpy_quantity_estimates import RMGSolvation
from bmapqml.chemxpl.minimized_functions.quantity_estimates import ConstrainedQuant
from bmapqml.chemxpl.minimized_functions.mol_constraints import NoProtonation
import numpy as np

SMILES_list = [
    "O",
    "N",
    "F",
    "Cl",
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


for solvent in ["water"]:
    print("SOLVENT:", solvent)
    solv_func = RMGSolvation(num_attempts=16, solvent_label=solvent)
    for SMILES in SMILES_list:
        tp = TrajectoryPoint(egc=SMILES_to_egc(SMILES))
        solv_en = solv_func(tp)
        print(SMILES, solv_en)
