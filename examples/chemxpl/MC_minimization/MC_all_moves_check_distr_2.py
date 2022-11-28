# The script verifies that Monte Carlo trajectories generated with RandomWalk objects follow the correct distribution when no bias is applied.
from bmapqml.chemxpl.valence_treatment import str2ChemGraph
from bmapqml.chemxpl.random_walk import RandomWalk, full_change_list
import random
import numpy as np
from bmapqml.chemxpl import ExtGraphCompound
from bmapqml.chemxpl.minimized_functions import NumHAtoms
from copy import deepcopy

random.seed(1)
np.random.seed(1)

# Constraints on chemical space - molecules with only C and P heavy atoms, maximum 4 heavy atoms large,
# bonds between two phosphorus atoms are forbidden.
possible_elements = ["C", "S", "O"]

forbidden_bonds = [(8, 8)]
not_protonated = [8]

max_nhatoms = 4

# Define the minimized function, which is in this case defined to return 0, 1, or 2 if the molecule contains 1 or 2, 3, or 4 heavy atoms.
intervals = [[1, 2], 3, 4]

min_func = NumHAtoms(intervals=intervals)

# The simulation is run for several replicas at once, each having a different beta value.
# If the beta value is set to "None", the corresponding replica undergoes greedy stochastic minimization of min_func;
# if global minimum of min_func is not unique all global minima should be represented with equal probability.
# For each real beta value the MC trajectory generates ensemble corresponding to probability density exp(-beta*min_func);
# these unsembles are used to decrease the probability that greedy optimization returns a local minimum rather than a global one.
ln2 = np.log(2.0)

betas = [None, None, 2.0 * ln2, ln2, ln2 / 2.0, ln2 / 4.0]

num_MC_steps = 40000  # 100000


randomized_change_params = {
    "max_fragment_num": 1,
    "nhatoms_range": [1, max_nhatoms],
    "final_nhatoms_range": [1, max_nhatoms],
    "possible_elements": possible_elements,
    "bond_order_changes": [-2, -1, 1, 2],
    "forbidden_bonds": forbidden_bonds,
    "added_bond_orders": [1, 2, 3],
    "not_protonated": not_protonated,
    "change_prob_dict": full_change_list,
}

# "simple" moves change one replica at a time, "genetic" make a genetic step, "tempering" does exchange same as parallel tempering.
global_change_params = {
    "num_parallel_tempering_tries": 5,
    "num_genetic_tries": 5,
    "prob_dict": {"simple": 0.5, "genetic": 0.25, "tempering": 0.25},
}

# Initial point for all replicas is CC#CC.
init_cg_str = "6#1@1:6@2:6@3:6#1"
# init_cg_str="6#4"

init_cg = str2ChemGraph(init_cg_str)

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
    linear_storage=True,
    make_restart_frequency=1000,
)
for MC_step in range(num_MC_steps):
    rw.global_random_change(**global_change_params)
    print(MC_step, rw.cur_tps)

# How to make a restart file in the end. (Unnecessary with make_restart_frequency set.)
rw.make_restart()

# The following is a short analysis for easy verification that the distribution is the correct one.
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

    # Cycle over all graphs visited during the simulation.
    for cur_tp in rw.histogram:
        cur_num_visits = cur_tp.visit_num(
            beta_id
        )  # visit_num(beta_id) gives number of times graph was visited by replica with id beta_id.
        val_id = min_func.int_output(cur_tp)
        if cur_num_visits != 0:
            visited_mol_nums[val_id] += 1
        mol_nums[val_id]
        distr_vals[val_id] += cur_num_visits
        distr2_vals[val_id] += cur_num_visits**2

    averages = []
    for val_id, (visited_mol_num, mol_num, distr, distr2) in enumerate(
        zip(visited_mol_nums, mol_nums, distr_vals, distr2_vals)
    ):
        av = float(distr) / mol_num
        av2 = float(distr2) / mol_num
        print("Function value:", val_id)
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
