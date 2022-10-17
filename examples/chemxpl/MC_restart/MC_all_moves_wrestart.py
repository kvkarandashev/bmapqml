from bmapqml.chemxpl.valence_treatment import str2ChemGraph
from bmapqml.chemxpl.random_walk import RandomWalk
import random, math
from bmapqml.chemxpl import ExtGraphCompound
from bmapqml.chemxpl.minimized_functions import OrderSlide, ConstrainedQuant
from bmapqml.chemxpl.minimized_functions.mol_constraints import NoProtonation
from copy import deepcopy
import numpy as np
import sys, os

num_MC_steps = int(sys.argv[1])

dump_restart_name = sys.argv[2]

init_restart_file = None

make_restart_frequency = None

if len(sys.argv) == 4:
    if sys.argv[3].startswith("init_restart_file="):
        init_restart_file = sys.argv[3].split("=")[1]
    if sys.argv[3].startswith("make_restart_frequency="):
        make_restart_frequency = int(sys.argv[3].split("=")[1])

possible_elements = ["C", "N", "O"]

forbidden_bonds = [(7, 7)]

no_protonation = [8]

ln2 = math.log(2.0)

# None corresponds to greedy optimization, other betas are used in a Metropolis scheme.
betas = [None, ln2, ln2 / 2]

bias_coeff = 1.0
bound_enforcing_coeff = 1.0

randomized_change_params = {
    "max_fragment_num": 1,
    "nhatoms_range": [2, 9],
    "final_nhatoms_range": [4, 7],
    "possible_elements": possible_elements,
    "bond_order_changes": [-1, 1],
    "forbidden_bonds": forbidden_bonds,
}
global_change_params = {
    "num_parallel_tempering_tries": 5,
    "num_genetic_tries": 5,
    "prob_dict": {"simple": 0.5, "genetic": 0.25, "tempering": 0.25},
}

# The initial compound in butane.
init_cg = str2ChemGraph("6#3@1:6#2@2:6#2@3:6#3")

negcs = len(betas)

init_egcs = [ExtGraphCompound(chemgraph=deepcopy(init_cg)) for _ in range(negcs)]

min_func_true = OrderSlide([6, 7, 8])

min_func_name = "OrderSlide"

constr = NoProtonation(no_protonation)

min_func_constr = ConstrainedQuant(min_func_true, min_func_name)

delete_temp_data = [min_func_name]  # None

rw = RandomWalk(
    bias_coeff=bias_coeff,
    randomized_change_params=randomized_change_params,
    bound_enforcing_coeff=bound_enforcing_coeff,
    betas=betas,
    min_function=min_func_constr,
    init_egcs=init_egcs,
    restart_file=dump_restart_name,
    make_restart_frequency=make_restart_frequency,
    keep_histogram=True,
    keep_full_trajectory=True,
    delete_temp_data=delete_temp_data,
)

if init_restart_file is None:
    # Init random number generator.
    random.seed(1)
    np.random.seed(1)
else:
    rw.restart_from(init_restart_file)


for MC_step in range(num_MC_steps):
    rw.global_random_change(**global_change_params)

mr_kwargs = {}
if make_restart_frequency is not None:
    mr_kwargs = {"restart_file": "final_restart.pkl"}
rw.make_restart(**mr_kwargs)

num_order_slides = 0

for entry in rw.histogram:
    if min_func_name in entry.calculated_data:
        num_order_slides += 1

print("Number of OrderSlide entries stored in histogram:", num_order_slides)
