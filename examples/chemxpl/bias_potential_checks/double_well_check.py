# Check that for a diatomic barrier the metadynamics-like potential eventually leads to hopping between two local potential minima.
from bmapqml.chemxpl.valence_treatment import str2ChemGraph
from bmapqml.chemxpl.random_walk import RandomWalk
import random
import numpy as np
from bmapqml.chemxpl import ExtGraphCompound
from bmapqml.chemxpl.minimized_functions import Diatomic_barrier
from copy import deepcopy

bias_coeff = 256

seed = 1

random.seed(seed)
np.random.seed(seed)

possible_elements = ["F", "Cl"]

min_func = Diatomic_barrier([9, 17])

randomized_change_params = {
    "nhatoms_range": [2, 2],
    "final_nhatoms_range": [2, 2],
    "possible_elements": possible_elements,
}


betas = [4096.0]

num_MC_steps = 200


global_change_params = {
    "prob_dict": {"simple": 1.0},
}

# Initial point for all replicas is fluorine.
init_cg = str2ChemGraph("9@1:9")

init_egcs = [ExtGraphCompound(chemgraph=init_cg)]

# Create objects that runs the simulation.

rw = RandomWalk(
    bias_coeff=bias_coeff,
    vbeta_bias_coeff=None,
    randomized_change_params=randomized_change_params,
    betas=betas,
    min_function=min_func,
    init_egcs=init_egcs,
    keep_histogram=True,
    keep_full_trajectory=True,
    restart_file="restart.pkl",
    track_histogram_size=True,
)
for MC_step in range(num_MC_steps):
    rw.global_random_change(**global_change_params)
    print(MC_step, rw.cur_tps)

rw.make_restart()
