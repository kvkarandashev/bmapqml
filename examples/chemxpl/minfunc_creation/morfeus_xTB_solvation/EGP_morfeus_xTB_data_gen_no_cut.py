from bmapqml.chemxpl.random_walk import TrajectoryPoint
from bmapqml.chemxpl.valence_treatment import str2ChemGraph
from bmapqml.chemxpl.minimized_functions.morfeus_quantity_estimates import (
    morfeus_FF_xTB_code_quants,
    gen_atom_energies,
)
from bmapqml.chemxpl.valence_treatment import InvalidAdjMat
import sys
from bmapqml.utils import dump2pkl, embarrassingly_parallel

solvent = sys.argv[1]

NPROCS = int(sys.argv[2])

num_conformers = 32
num_attempts = 4
remaining_rho = None #0.9

forcefield = "MMFF94"

# to make sure atom energies are initialized
gen_atom_energies(
    [1, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 35], solvent=solvent
)


def chemgraph_str_data_calc(chemgraph_str):
    print(chemgraph_str)
    output = {"chemgraph_str": chemgraph_str}
    cg = str2ChemGraph(chemgraph_str)
    try:
        tp = TrajectoryPoint(cg=cg)
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
    return {**calc_data, **output}


def all_chemgraph_str_data_calc(chemgraph_str_list):
    return embarrassingly_parallel(
        chemgraph_str_data_calc, chemgraph_str_list, (), num_procs=NPROCS
    )


# File created with scripts found in examples/dataset_processing/EGP
EGP_chemgraph_str_file = sys.argv[3]

with open(EGP_chemgraph_str_file) as f:
    chemgraph_str_list = f.readlines()

xyz_data = all_chemgraph_str_data_calc(chemgraph_str_list)

pkl_name = "morfeus_xTB_data_" + solvent + ".pkl"

dump2pkl(xyz_data, pkl_name)
