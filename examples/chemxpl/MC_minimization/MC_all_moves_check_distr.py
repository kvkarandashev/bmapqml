# The script uses a toy "Diatomic_barrier" potential (see class description for the idea behind it)
# to demonstrate how RandomWalk class can be used to Monte Carlo optimization.

from bmapqml.chemxpl.valence_treatment import str2ChemGraph
from bmapqml.chemxpl.random_walk import RandomWalk
import random
import numpy as np
from bmapqml.chemxpl import ExtGraphCompound
from bmapqml.chemxpl.minimized_functions import NumHAtoms
from copy import deepcopy

random.seed(1)
np.random.seed(1)

possible_elements = ["C", "P"]

forbidden_bonds = [(15, 15)]

ln2 = np.log(2.0)

# For each beta defined here RandomWalk would create a replica that
# These replicas interact with each other via parallel tempering and ``genetic tempering'' moves.
# None corresponds to a replica that undergoes greedy stochastic optimization.

max_nhatoms = 4

betas = [None, None, 2.0 * ln2, ln2, ln2 / 2.0]

num_MC_steps = 40000  # 20000

bias_coeff = None
bound_enforcing_coeff = None

randomized_change_params = {
    "max_fragment_num": 1,
    "nhatoms_range": [1, max_nhatoms],
    "final_nhatoms_range": [1, max_nhatoms],
    "possible_elements": possible_elements,
    "bond_order_changes": [-1, 1],
    "forbidden_bonds": forbidden_bonds,
}
global_change_params = {
    "num_parallel_tempering_tries": 5,
    "num_genetic_tries": 5,
    "prob_dict": {"simple": 0.5, "genetic": 0.25, "tempering": 0.25},
}

init_cg = str2ChemGraph("6#1@1:6@2:6@3:6#1")

negcs = len(betas)

init_egcs = [ExtGraphCompound(chemgraph=deepcopy(init_cg)) for _ in range(negcs)]

histogram = [[] for _ in range(negcs)]
histogram_labels = [[] for _ in range(negcs)]

intervals = [[1, 2], 3, 4]

min_func = NumHAtoms(intervals=intervals)

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
    linear_storage=True,
)
for MC_step in range(num_MC_steps):
    rw.global_random_change(**global_change_params)
    print(MC_step, rw.cur_tps)

rw.make_restart()

num_intervals = len(intervals)

mol_nums = np.zeros((num_intervals,), dtype=int)
for cur_tp in rw.histogram:
    val_id = min_func.int_output(cur_tp)
    mol_nums[val_id] += 1

print("Number of molecules considered:", mol_nums)

for beta_id, beta in enumerate(betas):
    print("\n\nBeta:", beta)
    distr_vals = np.zeros((num_intervals,), dtype=int)
    distr2_vals = np.zeros((num_intervals,), dtype=int)
    visited_mol_nums = np.zeros((num_intervals,), dtype=int)

    for cur_tp in rw.histogram:
        cur_num_visits = cur_tp.visit_num(beta_id)
        val_id = min_func.int_output(cur_tp)
        if cur_num_visits != 0:
            visited_mol_nums[val_id] += 1
        mol_nums[val_id]
        distr_vals[val_id] += cur_num_visits
        distr2_vals[val_id] += cur_num_visits**2

    averages = []
    for visited_mol_num, mol_num, distr, distr2 in zip(
        visited_mol_nums, mol_nums, distr_vals, distr2_vals
    ):
        av = float(distr) / mol_num
        av2 = float(distr2) / mol_num
        print("Number of visited molecules:", visited_mol_num)
        print("Average:", av)
        print("Standard deviation:", np.sqrt((av2 - av**2) / mol_num))
        averages.append(av)
    if beta is not None:
        tot_log_deviation = 0.0
        for av_id, av in enumerate(averages):
            for other_av_id, other_av in enumerate(averages[:av_id]):
                if (
                    (distr_vals[av_id] != 0)
                    and (distr_vals[other_av_id] != 0)
                    and (tot_log_deviation is not None)
                ):
                    tot_log_deviation += (
                        np.log(av / other_av) + beta * (av_id - other_av_id)
                    ) ** 2
                else:
                    tot_log_deviation = None
        print("Hist from pot diff deviation:", tot_log_deviation)
