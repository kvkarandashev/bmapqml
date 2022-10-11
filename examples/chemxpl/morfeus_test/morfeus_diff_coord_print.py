from bmapqml.chemxpl.minimized_functions.morfeus_quantity_estimates import (
    morfeus_coord_info_from_tp,
)
from bmapqml.chemxpl.utils import SMILES_to_egc
from bmapqml.chemxpl.random_walk import TrajectoryPoint
from bmapqml.utils import write_xyz_file
import sys

SMILES = sys.argv[1]

num_attempts = int(sys.argv[2])

if len(sys.argv) < 4:
    xyz_prefac = ""
else:
    xyz_prefac = sys.argv[3]

tp = TrajectoryPoint(egc=SMILES_to_egc(SMILES))

coord_info = morfeus_coord_info_from_tp(tp, num_attempts=num_attempts, all_confs=True)

for i, (coords, energy, degeneracy, weight) in enumerate(
    zip(
        coord_info["coordinates"],
        coord_info["rdkit_energy"],
        coord_info["rdkit_degeneracies"],
        coord_info["rdkit_Boltzmann"],
    )
):
    xyz_name = xyz_prefac + str(i) + ".xyz"
    write_xyz_file(
        coords,
        xyz_name,
        nuclear_charges=coord_info["nuclear_charges"],
        extra_string="energy="
        + str(energy)
        + " degen="
        + str(degeneracy)
        + " weight="
        + str(weight),
    )
