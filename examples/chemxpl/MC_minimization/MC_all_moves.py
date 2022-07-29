# The script uses a toy "Diatomic_barrier" potential (see class description for the idea behind it)
# to demonstrate how RandomWalk class can be used to Monte Carlo optimization.

from bmapqml.chemxpl.valence_treatment import ChemGraph
from bmapqml.chemxpl.random_walk import RandomWalk
import random, math
from bmapqml.chemxpl import ExtGraphCompound
from bmapqml.chemxpl.minimized_functions import Diatomic_barrier
from copy import deepcopy

random.seed(1)

possible_elements = ["Cl", "F"]

forbidden_bonds = None  # [(17, 9)]

ln2 = math.log(2.0)

# For each beta defined here RandomWalk would create a replica that
# These replicas interact with each other via parallel tempering and ``genetic temperting'' moves.
# None corresponds to a replica that undergoes greedy stochastic optimization.

betas = [None, ln2, ln2 / 2]

num_MC_steps = 100000  # 100000

bias_coeff = None
bound_enforcing_coeff = None

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

# All replicas will start in one position that in chemical space corresponds to the local minimum.
# Note that the "greedy optimization" algorithm wouldn't be able to get to the global minimum on its own
# as it is separated from the initial position by a ``transition state'' in chemical space.
init_egcs = [ExtGraphCompound(chemgraph=deepcopy(init_cg)) for _ in range(negcs)]

histogram = [[] for i in range(negcs)]
histogram_labels = [[] for i in range(negcs)]

min_func = Diatomic_barrier([9, 17])

rw = RandomWalk(
    bias_coeff=bias_coeff,
    randomized_change_params=randomized_change_params,
    bound_enforcing_coeff=bound_enforcing_coeff,
    betas=betas,
    min_function=min_func,
    init_egcs=init_egcs,
)
for MC_step in range(num_MC_steps):
    # If changing randomized_change_params is required mid-simulation they can be updated via *.change_rdkit arguments
    rw.global_random_change(**global_change_params)
    cur_egcs = [tp.egc for tp in rw.cur_tps]
    for i in range(negcs):
        cur_egc = cur_egcs[i]
        if cur_egc not in histogram_labels[i]:
            histogram_labels[i].append(deepcopy(cur_egc))
            histogram[i].append(0)
        cur_egc_id = histogram_labels[i].index(cur_egc)
        histogram[i][cur_egc_id] += 1
    print(MC_step, cur_egcs)

global_hist_labels = []
global_histogram = []

# Print histograms corresponding to each of the betas. If everything went correctly,
# None should've spent overwhelming part of the simulation in the global minimum ("F2" or "9@1:9" as the code outputs it),
# for the others the ratio of probabilities of F2, Cl2 ("17@1:17"), and ClF ("17@1:9") should approach 1.0 - exp(-beta) - exp(-2*beta) as the number of Monte Carlo steps
# increases.
for i, beta in enumerate(betas):
    print("Beta:", beta, "index:", i)
    for egc, distr in zip(histogram_labels[i], histogram[i]):
        print("EGC:", egc)
        print("Distribution:", distr)
    for hl, hv in zip(histogram_labels[i], histogram[i]):
        if hl not in global_hist_labels:
            global_hist_labels.append(hl)
            global_histogram.append(0)
        i = global_hist_labels.index(hl)
        global_histogram[i] += hv

print("Global histogram:")
for egc, distr in zip(global_hist_labels, global_histogram):
    print(egc, distr)
