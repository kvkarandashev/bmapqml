# Sanity check demonstrating that there is no dimensionality discrepancy between Morfeus and xTB.
from bmapqml.chemxpl.utils import SMILES_to_egc
from bmapqml.chemxpl.random_walk import TrajectoryPoint
from bmapqml.chemxpl.minimized_functions.morfeus_quantity_estimates import (
    morfeus_coord_info_from_tp,
    xTB_quants,
    morfeus_FF_xTB_code_quants,
)
import numpy as np
from bmapqml.data import conversion_coefficient

SMILES = "F"
tp = TrajectoryPoint(egc=SMILES_to_egc(SMILES))

print(tp)

coord_info = morfeus_coord_info_from_tp(tp, num_attempts=4)

coordinates = coord_info["coordinates"]

dist = np.sqrt(np.sum((coordinates[0] - coordinates[1]) ** 2))

print("MMFF94 distance:", dist * conversion_coefficient["Angstrom_Bohr"])

cur_dist = dist / 2.0

npoints = 2000

ddist = 0.001

ncharges = np.array([9, 1])

prev_en = None

quantities = ["energy", "HOMO_LUMO_gap", "dipole"]

print("Morfeus+xTB results")

print(morfeus_FF_xTB_code_quants(tp, num_attempts=4, quantities=quantities))

print("Brute xTB results")

print("Distance (Bohr) # energy # HOMO-LUMO gap # Dipole")

for i in range(npoints):
    coordinates = np.array([[cur_dist, 0.0, 0.0], [0.0, 0.0, 0.0]])
    try:
        q = xTB_quants(coordinates, ncharges, quantities=quantities)
    except:
        quit()
    en = q["energy"]
    if (prev_en is not None) and (en > prev_en):
        print(cur_dist, en, q["HOMO_LUMO_gap"], q["dipole"])
        quit()
    prev_en = en
    cur_dist += ddist
