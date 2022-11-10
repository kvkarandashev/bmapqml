# Enumerate chemical graphs corresponding to some constraints.
from bmapqml.chemxpl.valence_treatment import str2ChemGraph
from bmapqml.chemxpl.random_walk import RandomWalk, full_change_list
from bmapqml.chemxpl.modify import (
    egc_valid_wrt_change_params,
    change_bond_order,
    replace_heavy_atom,
)
import random, sys
import numpy as np
from bmapqml.chemxpl import ExtGraphCompound
from bmapqml.chemxpl.minimized_functions import NumHAtoms
from copy import deepcopy

random.seed(1)
np.random.seed(1)

#
possible_elements = ["C", "S", "O", "F"]

max_nhatoms = 4

min_nhatoms = 1

basic_constraints = {
    "change_prob_dict": full_change_list,
    "nhatoms_range": [min_nhatoms, max_nhatoms],
    "final_nhatoms_range": [min_nhatoms, max_nhatoms],
    "possible_elements": possible_elements,
}

# "forbidden_bonds" and "not_protonated" are two types of constraints that can be forced explicitly in the sampling procedure.
advanced_constraints = {
    "forbidden_bonds": [(8, 8), (8, 9), (9, 9), (16, 16)],
    "not_protonated": [8, 9, 16],
}

# Decide whether "other_constraints" will be enforced during initial histogram generation.
if len(sys.argv) > 1:
    IMPLICIT_CONSTRAINT = sys.argv[1] == "TRUE"
else:
    IMPLICIT_CONSTRAINT = False

# Default random change parameters.

randomized_change_params = {
    **basic_constraints,
    "max_fragment_num": 1,
    "possible_elements": possible_elements,
    "bond_order_changes": [-1, 1],
    "added_bond_orders": [1, 2, 3],
}


if IMPLICIT_CONSTRAINT:
    randomized_change_params = {**randomized_change_params, **advanced_constraints}

intervals = [[1, 2], 3, 4]

min_func = NumHAtoms(intervals=intervals)

ln2 = np.log(2.0)

betas = [None, None, 2.0 * ln2, ln2, ln2 / 2.0, ln2 / 2.0, ln2 / 4.0, ln2 / 4.0]

num_MC_steps = 50000  # 100000

# "simple" moves change one replica at a time, "genetic" make a genetic step, "tempering" does exchange same as parallel tempering.
global_change_params = {
    "num_parallel_tempering_tries": 5,
    "num_genetic_tries": 5,
    "prob_dict": {"simple": 0.5, "genetic": 0.25, "tempering": 0.25},
}

# Initial point for all replicas is methane.
init_cg = str2ChemGraph("6#4")

init_egcs = [ExtGraphCompound(chemgraph=deepcopy(init_cg)) for _ in range(len(betas))]

# Create objects that runs the simulation.

rw = RandomWalk(
    bias_coeff=None,
    vbeta_bias_coeff=None,
    randomized_change_params=randomized_change_params,
    betas=betas,
    min_function=min_func,
    init_egcs=init_egcs,
    keep_histogram=True,
    keep_full_trajectory=True,
    restart_file="restart.pkl",
    make_restart_frequency=5000,
)
for MC_step in range(num_MC_steps):
    rw.global_random_change(**global_change_params)
#    print(MC_step, rw.cur_tps)

# How to make a restart file in the end. (Unnecessary with make_restart_frequency set.)
rw.make_restart()

found_nums_unconstr = np.zeros((max_nhatoms - min_nhatoms + 1,), dtype=int)

found_nums_constr = np.zeros((max_nhatoms - min_nhatoms + 1,), dtype=int)

for entry in rw.histogram:
    cur_egc = entry.egc
    nha = cur_egc.num_heavy_atoms()
    if not egc_valid_wrt_change_params(cur_egc, **basic_constraints):
        print("Basic constraints violated for ", cur_egc)
        quit()
    nha_id = nha - min_nhatoms
    found_nums_unconstr[nha_id] += 1
    if egc_valid_wrt_change_params(
        cur_egc, **basic_constraints, **advanced_constraints
    ):
        found_nums_constr[nha_id] += 1

print(
    "Number of heavy atoms : found chemgraphs : found chemgraphs satisfying advanced constraints."
)
for nha_id, (nunconstr, nconstr) in enumerate(
    zip(found_nums_unconstr, found_nums_constr)
):
    print(nha_id + 1, ":", nunconstr, ":", nconstr)
