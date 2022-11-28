# The script uses a toy "Diatomic_barrier" potential (see class description for the idea behind it)
# to demonstrate how RandomWalk class can be used to Monte Carlo optimization.

from bmapqml.chemxpl.valence_treatment import ChemGraph
from bmapqml.chemxpl.random_walk import RandomWalk, gen_exp_beta_array
import random
import numpy as np
from bmapqml.chemxpl import ExtGraphCompound
from bmapqml.chemxpl.minimized_functions import OrderSlide
from copy import deepcopy

random.seed(2)
np.random.seed(2)

possible_elements = ["C", "N", "O", "F"]

forbidden_bonds = [(7, 7), (8, 8), (9, 9), (7, 8), (7, 9), (8, 9)]

# For each beta defined here RandomWalk would create a replica that
# These replicas interact with each other via parallel tempering and ``genetic tempering'' moves.
# None corresponds to a replica that undergoes greedy stochastic optimization.

betas = gen_exp_beta_array(4, 1.0, 4, max_real_beta=4.0)

num_MC_steps = 10000  # 100000

bias_coeff = None
bound_enforcing_coeff = None

randomized_change_params = {
    "max_fragment_num": 1,
    "nhatoms_range": [1, 9],
    "final_nhatoms_range": [1, 9],
    "possible_elements": possible_elements,
    "bond_order_changes": [-1, 1],
    "forbidden_bonds": forbidden_bonds,
}
global_change_params = {
    "num_genetic_tries": 5,
    "num_tempering_tries": 5,
    "prob_dict": {"simple": 0.75, "genetic": 0.25},
}

init_ncharges = [6]
init_bond_orders = {(0, 1): 1}

init_cg = ChemGraph(
    nuclear_charges=init_ncharges, bond_orders=init_bond_orders, hydrogen_autofill=True
)

negcs = len(betas)

init_egcs = [ExtGraphCompound(chemgraph=deepcopy(init_cg)) for _ in range(negcs)]

histogram = [[] for _ in range(negcs)]
histogram_labels = [[] for _ in range(negcs)]

min_func = OrderSlide(possible_elements=possible_elements)

rw = RandomWalk(
    bias_coeff=bias_coeff,
    randomized_change_params=randomized_change_params,
    bound_enforcing_coeff=bound_enforcing_coeff,
    betas=betas,
    min_function=min_func,
    init_egcs=init_egcs,
    keep_histogram=True,
    keep_full_trajectory=True,
    restart_file="larger_mols_restart.pkl",
    greedy_delete_checked_paths=True,
)
for MC_step in range(num_MC_steps):
    rw.global_random_change(**global_change_params)
    print(MC_step, rw.cur_tps)

rw.make_restart()
