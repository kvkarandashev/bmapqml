from bmapqml.test_utils import dirs_xyz_list
from bmapqml.chemxpl.utils import xyz2mol_extgraph
from bmapqml.chemxpl.random_walk import TrajectoryPoint
from bmapqml.chemxpl.minimized_functions.morfeus_quantity_estimates import (
    morfeus_FF_xTB_code_quants,
)
from bmapqml.chemxpl.valence_treatment import InvalidAdjMat
import os, sys
from bmapqml.utils import dump2pkl, embarrassingly_parallel

solvent = sys.argv[1]

NPROCS = int(sys.argv[2])

num_conformers = 50
num_attempts = 4

forcefield = "MMFF94"


def xyz_data_extract(xyz_name):
    output = {"xyz": xyz_name}
    try:
        tp = TrajectoryPoint(egc=xyz2mol_extgraph(xyz_name))
    except InvalidAdjMat:
        return output
    calc_data = morfeus_FF_xTB_code_quants(
        tp,
        num_conformers=num_conformers,
        num_attempts=num_attempts,
        ff_type=forcefield,
        quantities=["total_energy", "gap", "solvation_energy"],
        solvent=solvent,
    )
    return {**calc_data, **output, "chemgraph": str(tp)}


def all_xyzs_data_extract(xyz_list):
    return embarrassingly_parallel(xyz_data_extract, xyz_list, (), num_procs=NPROCS)


QM9_xyz_dir = os.environ["DATA"] + "/QM9_formatted"

QM9_xyzs = sorted(dirs_xyz_list(QM9_xyz_dir))

xyz_data = all_xyzs_data_extract(QM9_xyzs)

pkl_name = "morfeus_xTB_data_" + solvent + ".pkl"

dump2pkl(xyz_data, pkl_name)
