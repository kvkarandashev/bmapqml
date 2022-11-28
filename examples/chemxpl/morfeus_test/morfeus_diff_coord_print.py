from bmapqml.chemxpl.minimized_functions.morfeus_quantity_estimates import (
    morfeus_coord_info_from_tp,
)
from bmapqml.chemxpl.utils import SMILES_to_egc
from bmapqml.chemxpl.random_walk import TrajectoryPoint
from bmapqml.chemxpl.ext_graph_compound import ExtGraphCompound
from bmapqml.chemxpl.valence_treatment import str2ChemGraph
from bmapqml.utils import write_xyz_file
import sys

if sys.argv[1] == "--chemgraphstr":
    nflags = 1
    egc = ExtGraphCompound(chemgraph=str2ChemGraph(sys.argv[1 + nflags]))
else:
    nflags = 0
    SMILES = sys.argv[1 + nflags]
    egc = SMILES_to_egc(SMILES)

num_attempts = int(sys.argv[2 + nflags])

if len(sys.argv) < 4 + nflags:
    xyz_prefac = ""
else:
    xyz_prefac = sys.argv[3 + nflags]

tp = TrajectoryPoint(egc=egc)

coord_info = morfeus_coord_info_from_tp(tp, num_attempts=num_attempts, all_confs=True)

all_coords = coord_info["coordinates"]

for i, coords in enumerate(all_coords):
    xyz_name = xyz_prefac + str(i) + ".xyz"
    extra_string = ""
    for printed_quant in ["rdkit_energy", "rdkit_degeneracy", "rdkit_Boltzmann"]:
        extra_string += " " + printed_quant + "=" + str(coord_info[printed_quant][i])
    write_xyz_file(
        coords,
        xyz_name,
        nuclear_charges=coord_info["nuclear_charges"],
        extra_string=extra_string[1:],
    )
