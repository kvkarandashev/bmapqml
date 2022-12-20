from bmapqml.test_utils import dirs_xyz_list
from bmapqml.chemxpl.utils import xyz2mol_extgraph, SMILES_to_egc
from bmapqml.chemxpl.random_walk import TrajectoryPoint
from bmapqml.chemxpl.minimized_functions.morfeus_quantity_estimates import (
    morfeus_FF_xTB_code_quants,
    gen_atom_energies,
)
from bmapqml.chemxpl.valence_treatment import InvalidAdjMat
import os, sys
from bmapqml.utils import dump2pkl, embarrassingly_parallel

solvent = sys.argv[1]

NPROCS = int(sys.argv[2])

num_conformers = 32
num_attempts = 4
remaining_rho = None #0.9

forcefield = "MMFF94"

# to make sure atom energies are initialized
gen_atom_energies([1, 6, 7, 8, 9], solvent=solvent)


def xyz_data_extract(xyz_name):
    print(xyz_name)
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
        remaining_rho=remaining_rho,
        quantities=[
            "dipole",
            "energy",
            "HOMO_LUMO_gap",
            "solvation_energy",
            "energy_no_solvent",
            "atomization_energy",
            "normalized_atomization_energy",
            "num_evals",
        ],
        solvent=solvent,
    )
    return {**calc_data, **output, "chemgraph": str(tp)}


def all_xyzs_data_extract(xyz_list):
    return embarrassingly_parallel(xyz_data_extract, xyz_list, (), num_procs=NPROCS)


QM9_xyz_dir = os.environ["DATA"] + "/QM9_formatted"

QM9_xyzs = dirs_xyz_list(QM9_xyz_dir)

xyz_data = all_xyzs_data_extract(QM9_xyzs)

pkl_name = "morfeus_xTB_data_" + solvent + ".pkl"

dump2pkl(xyz_data, pkl_name)
