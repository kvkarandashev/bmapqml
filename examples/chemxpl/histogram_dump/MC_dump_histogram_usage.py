from bmapqml.chemxpl.valence_treatment import str2ChemGraph
from bmapqml.chemxpl.random_walk import RandomWalk
import random, sys
from bmapqml.chemxpl import ExtGraphCompound
from bmapqml.chemxpl.minimized_functions import ChargeSum
from copy import deepcopy
import numpy as np

random.seed(1)
np.random.seed(1)

restart_file_name = sys.argv[1]

if len(sys.argv) > 2:
    histogram_dump_file_prefix = sys.argv[2]
    max_histogram_size = 500
else:
    histogram_dump_file_prefix = None
    max_histogram_size = None

num_MC_steps = 20000

possible_elements = ["C", "N", "O"]

forbidden_bonds = [(7, 7), (8, 8)]

not_protonated = [8]

ln2 = np.log(2.0)

# None corresponds to greedy optimization, other betas are used in a Metropolis scheme.
betas = [None, 2 * ln2, ln2, ln2 / 2, ln2 / 4]

nhatoms_range = [1, 9]

init_cg_str = "6#3@1:6#2@2:6#2@3:6#3"

randomized_change_params = {
    "max_fragment_num": 1,
    "nhatoms_range": nhatoms_range,
    "final_nhatoms_range": nhatoms_range,
    "possible_elements": possible_elements,
    "bond_order_changes": [-1, 1],
    "forbidden_bonds": forbidden_bonds,
    "not_protonated": not_protonated,
}
global_change_params = {
    "num_parallel_tempering_tries": 5,
    "num_genetic_tries": 5,
    "prob_dict": {"simple": 0.5, "genetic": 0.25, "tempering": 0.25},
}

# The initial compound in butane.
init_cg = str2ChemGraph(init_cg_str)

negcs = len(betas)

init_egcs = [ExtGraphCompound(chemgraph=deepcopy(init_cg)) for _ in range(negcs)]

rw = RandomWalk(
    randomized_change_params=randomized_change_params,
    betas=betas,
    min_function=ChargeSum(),
    min_function_name="ChargeSum",
    init_egcs=init_egcs,
    restart_file=restart_file_name,
    max_histogram_size=max_histogram_size,
    histogram_dump_file_prefix=histogram_dump_file_prefix,
    track_histogram_size=True,
    keep_histogram=True,
    keep_full_trajectory=True,
    canonize_trajectory_points=True,  # important for reproducability
)


for MC_step in range(num_MC_steps):
    rw.global_random_change(**global_change_params)

mr_kwargs = {}
rw.make_restart()
