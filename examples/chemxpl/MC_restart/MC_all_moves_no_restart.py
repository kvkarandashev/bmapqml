# Does same as MC_all_moves_no_restart.py, but restarting mid-simulation.

from bmapqml.chemxpl.valence_treatment import ChemGraph
from bmapqml.chemxpl.random_walk import RandomWalk
import random, math
from bmapqml.chemxpl import ExtGraphCompound
from bmapqml.chemxpl.minimized_functions import Diatomic_barrier
from copy import deepcopy
import numpy as np

random.seed(1)
np.random.seed(1)

possible_elements = ["Cl", "F"]

forbidden_bonds = None  # [(17, 9)]

ln2 = math.log(2.0)

# None corresponds to greedy optimization, other betas are used in a Metropolis scheme.
betas = [None, ln2, ln2 / 2]
# betas=[ln2, ln2, ln2, ln2, ln2, ln2]

num_MC_steps = 100  # 100000

bias_coeff = None
bound_enforcing_coeff = 1.0

randomized_change_params = {
    "max_fragment_num": 1,
    "nhatoms_range": [2, 2],
    "final_nhatoms_range": [2, 2],
    "possible_elements": possible_elements,
    "bond_order_changes": [-1, 1],
    "forbidden_bonds": forbidden_bonds,
}
global_change_params = {
    "num_parallel_tempering_tries": 5,
    "num_genetic_tries": 5,
    "prob_dict": {"simple": 0.5, "genetic": 0.25, "tempering": 0.25},
}

init_ncharges = [17, 17]
init_bond_orders = {(0, 1): 1}

init_cg = ChemGraph(
    nuclear_charges=init_ncharges, bond_orders=init_bond_orders, hydrogen_autofill=True
)

negcs = len(betas)

init_egcs = [ExtGraphCompound(chemgraph=deepcopy(init_cg)) for i in range(negcs)]

histogram = [[] for i in range(negcs)]
histogram_labels = [[] for i in range(negcs)]

min_func = Diatomic_barrier([9, 17])

restart_file = "restart.pkl"

rw = RandomWalk(
    bias_coeff=bias_coeff,
    randomized_change_params=randomized_change_params,
    bound_enforcing_coeff=bound_enforcing_coeff,
    betas=betas,
    min_function=min_func,
    init_egcs=init_egcs,
    restart_file=restart_file,
    keep_histogram=True,
)
other_rw = deepcopy(rw)

for MC_step in range(num_MC_steps * 2):
    rw.global_random_change(**global_change_params)

for tp in rw.histogram:
    print(tp.egc)
    print(tp.num_visits)
