from bmapqml.chemxpl.valence_treatment import str2ChemGraph
from bmapqml.chemxpl.random_walk import RandomWalk
import random, math
from bmapqml.chemxpl import ExtGraphCompound
from bmapqml.chemxpl.minimized_functions import OrderSlide
from copy import deepcopy
import numpy as np
import sys, os

num_MC_steps = int(sys.argv[1])

dump_restart_name = sys.argv[2]

init_restart_file = None

if len(sys.argv) == 4:
    init_restart_file = sys.argv[3]

possible_elements = ["C", "N"]

forbidden_bonds = [(7, 7)]

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

min_func = OrderSlide([6, 7])

rw = RandomWalk(
    bias_coeff=bias_coeff,
    randomized_change_params=randomized_change_params,
    bound_enforcing_coeff=bound_enforcing_coeff,
    betas=betas,
    min_function=min_func,
    init_egcs=init_egcs,
    restart_file=dump_restart_name,
    keep_histogram=True,
    keep_full_trajectory=True,
)

if init_restart_file is None:
    # Init random number generator.
    random.seed(1)
    np.random.seed(1)
else:
    rw.restart_from(init_restart_file)


for MC_step in range(num_MC_steps):
    rw.global_random_change(**global_change_params)

rw.make_restart()
