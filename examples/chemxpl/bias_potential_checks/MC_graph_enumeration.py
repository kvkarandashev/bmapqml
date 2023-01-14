# Enumerate chemical graphs corresponding to some constraints.
from bmapqml.chemxpl.valence_treatment import str2ChemGraph
from bmapqml.chemxpl.random_walk import RandomWalk
from bmapqml.chemxpl.modify import egc_valid_wrt_change_params
import random, sys
import numpy as np
from bmapqml.chemxpl import ExtGraphCompound
from bmapqml.chemxpl.minimized_functions import ZeroFunc
from copy import deepcopy

bias_coeffs = {"none": None, "weak": 0.2, "stronger": 0.4}

# To use or not to use genetic moves.
if len(sys.argv) > 5:
    USE_GENETIC = sys.argv[5] == "TRUE"
else:
    USE_GENETIC = True

# Which biasing coefficient to use.
if len(sys.argv) > 4:
    bias_type = sys.argv[4]
else:
    bias_type = "none"  # "none"

# Decide whether constraints on covalent bonds will be enforced during initial histogram generation.
if len(sys.argv) > 3:
    IMPLICIT_CONSTRAINT = sys.argv[3] == "TRUE"
else:
    IMPLICIT_CONSTRAINT = False

# Choose size constraint
if len(sys.argv) > 2:
    max_nhatoms = int(sys.argv[2])
else:
    max_nhatoms = 6

# Choose seed.
if len(sys.argv) > 1:
    seed = int(sys.argv[1])
else:
    seed = 1

random.seed(seed)
np.random.seed(seed)

bias_coeff = bias_coeffs[bias_type]

possible_elements = ["C", "S", "P"]


basic_constraints = {
    "nhatoms_range": [1, max_nhatoms],
    "final_nhatoms_range": [1, max_nhatoms],
    "possible_elements": possible_elements,
}

# "forbidden_bonds" and "not_protonated" are two types of constraints that can be forced explicitly in the sampling procedure.
advanced_constraints = {
    "forbidden_bonds": [(15, 15), (16, 16)],
    "not_protonated": [
        16
    ],  # [15, 16], leaving one polyvalent as eligible for protonation to test hydrogenation+valence change
}

# Default random change parameters.

randomized_change_params = {
    **basic_constraints,
    "max_fragment_num": 1,
    "bond_order_changes": [-1, 1],
    "added_bond_orders": [1, 2, 3],
}

if IMPLICIT_CONSTRAINT:
    randomized_change_params = {**randomized_change_params, **advanced_constraints}

betas = [1.0 for _ in range(36)]

num_MC_steps = 50000  # 50000

# "simple" moves change one replica at a time, "genetic" make a genetic step.
if USE_GENETIC:
    genetic_proportion = 0.25
else:
    genetic_proportion = 0.0

global_change_params = {
    "num_genetic_tries": 32,
    "prob_dict": {"simple": 0.75, "genetic": genetic_proportion},
}

# Initial point for all replicas is methane.
init_cg = str2ChemGraph("6#4")

init_egcs = [ExtGraphCompound(chemgraph=deepcopy(init_cg)) for _ in range(len(betas))]

# Create objects that runs the simulation.
cur_maximum = None
max_reached = None

rw = RandomWalk(
    bias_coeff=bias_coeff,
    vbeta_bias_coeff=None,
    randomized_change_params=randomized_change_params,
    betas=betas,
    min_function=ZeroFunc(),
    init_egcs=init_egcs,
    keep_histogram=True,
    keep_full_trajectory=True,
    restart_file="restart.pkl",
    make_restart_frequency=1000,
    track_histogram_size=True,
    debug=True,
)

for MC_step in range(num_MC_steps):
    rw.global_random_change(**global_change_params)
    cur_hist_size = len(rw.histogram)
    if (cur_maximum is None) or (cur_hist_size > cur_maximum):
        cur_maximum = cur_hist_size
        max_reached = MC_step

# Make a restart file in the end.
rw.make_restart()

print("Total number of molecules found:", len(rw.histogram))

found_mols_satisfying_constraints = np.zeros((max_nhatoms,), dtype=int)
found_mols = np.zeros((max_nhatoms,), dtype=int)
for entry in rw.histogram:
    cur_egc = entry.egc
    if not egc_valid_wrt_change_params(cur_egc, **basic_constraints):
        print("Basic constraints violated for ", cur_egc)
        quit()
    nha_id = cur_egc.num_heavy_atoms() - 1
    found_mols[nha_id] += 1
    if egc_valid_wrt_change_params(
        cur_egc, **basic_constraints, **advanced_constraints
    ):
        found_mols_satisfying_constraints[nha_id] += 1

print(
    "Number of heavy atoms : found chemgraphs : found chemgraphs satisfying advanced constraints."
)
print("MC step when maximum was reached:", max_reached)
for nha_id, (ntot, nconstr) in enumerate(
    zip(found_mols, found_mols_satisfying_constraints)
):
    print(nha_id + 1, ":", ntot, ":", nconstr)
