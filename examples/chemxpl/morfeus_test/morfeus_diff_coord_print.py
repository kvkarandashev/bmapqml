from bmapqml.chemxpl.minimized_functions.morfeus_quantity_estimates import morfeus_coord_info_from_tp
from bmapqml.chemxpl.utils import SMILES_to_egc
from bmapqml.chemxpl.random_walk import TrajectoryPoint

SMILES="FCCF"

num_attempts=2

tp=TrajectoryPoint(egc=SMILES_to_egc(SMILES))

print("Morfeus coordinates:", morfeus_coord_info_from_tp(tp, num_attempts=num_attempts))

print("Morfeus all conf coordinates:", morfeus_coord_info_from_tp(tp, num_attempts=num_attempts, all_confs=True))
