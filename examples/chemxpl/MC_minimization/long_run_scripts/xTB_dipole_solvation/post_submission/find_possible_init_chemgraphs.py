# Takes a folder with *.xyz files and

import numpy as np
from glob import glob
from bmapqml.utils import loadpkl, dump2pkl, embarrassingly_parallel
from bmapqml.chemxpl.utils import xyz2mol_extgraph
from bmapqml.chemxpl.ext_graph_compound import ExtGraphCompound
from bmapqml.chemxpl.valence_treatment import InvalidAdjMat
from bmapqml.chemxpl.random_walk import egc_valid_wrt_change_params, TrajectoryPoint
import sys

gap_name = "HOMO_LUMO_gap"


def xyz2tp(xyz_name):
    print("#", xyz_name)
    try:
        egc = xyz2mol_extgraph(xyz_name)
        return TrajectoryPoint(egc=egc)
    except InvalidAdjMat:
        return None


def all_xyzs2tps(xyz_list):
    return embarrassingly_parallel(xyz2tp, xyz_list, ())


def compliant_tp_list(tps, **other_constraints):
    output = []
    for tp in tps:
        if tp is None:
            continue
        if egc_valid_wrt_change_params(tp.egc, **other_constraints):
            output.append(tp)
    return output


# Script options.
xyz_dir = sys.argv[1]
dataset = sys.argv[2]
quant_func_pkl = sys.argv[3]

# Import all xyzs.
all_tps = all_xyzs2tps(glob(xyz_dir + "/*.xyz"))


# Set the chemical space constraints
if dataset == "qm9":
    possible_elements = ["C", "N", "O", "F"]
    nhatoms_range = [1, 9]
    not_protonated = [8, 9]
    forbidden_bonds = (
        [
            (7, 7),
            (8, 8),
            (9, 9),
            (7, 8),
            (7, 9),
            (8, 9),
        ],
    )
else:
    possible_elements = ["B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br"]
    nhatoms_range = [1, 15]
    not_protonated = [5, 8, 9, 14, 15, 16, 17, 35]
    forbidden_bonds = [
        (7, 7),
        (7, 8),
        (8, 8),
        (7, 9),
        (8, 9),
        (9, 9),
        (7, 17),
        (8, 17),
        (9, 17),
        (17, 17),
        (7, 35),
        (8, 35),
        (9, 35),
        (17, 35),
        (35, 35),
        (15, 15),
        (16, 16),
    ]
constraints = {
    "nhatoms_range": nhatoms_range,
    "not_protonated": not_protonated,
    "forbidden_bonds": forbidden_bonds,
}

# Print list of ChemGraphs that satisfy constraints on the chemical space.
compliant_tps = compliant_tp_list(all_tps, **constraints)

compliant_txt_name = dataset + "_compliant_str"

with open(compliant_txt_name + ".txt", "w") as f:
    for tp in compliant_tps:
        print(tp, file=f)

# Find for which graphs the optimized quantity does not return None.
quant_func = loadpkl(quant_func_pkl)

compliant_tp_quants = embarrassingly_parallel(quant_func, compliant_tps, ())

with open(compliant_txt_name + "_valid_evals.txt", "w") as f:
    for quant, tp in zip(compliant_tp_quants, compliant_tps):
        if quant is not None:
            print(tp, file=f)
