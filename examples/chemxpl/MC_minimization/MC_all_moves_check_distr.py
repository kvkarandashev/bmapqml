# The script verifies that Monte Carlo trajectories generated with RandomWalk objects follow the correct distribution when no bias is applied.
from bmapqml.chemxpl.valence_treatment import str2ChemGraph
from bmapqml.chemxpl.random_walk import RandomWalk
import random
import numpy as np
from bmapqml.chemxpl import ExtGraphCompound
from bmapqml.chemxpl.minimized_functions import NumHAtoms
from copy import deepcopy
from bmapqml.chemxpl.test_utils import print_distribution_analysis

random.seed(1)
np.random.seed(1)

# Constraints on chemical space - molecules with only C and P heavy atoms, maximum 4 heavy atoms large,
# bonds between two phosphorus atoms are forbidden.
possible_elements = ["C", "P"]

forbidden_bonds = [(15, 15)]

not_protonated = [15]

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

betas = [None, 2.0 * ln2, ln2, ln2 / 2.0]

num_MC_steps = 10000  # 10000


randomized_change_params = {
    "max_fragment_num": 1,
    "nhatoms_range": [1, max_nhatoms],
    "final_nhatoms_range": [1, max_nhatoms],
    "possible_elements": possible_elements,
    "bond_order_changes": [-1, 1],
    "forbidden_bonds": forbidden_bonds,
    "cross_coupling_smallest_exchange_size": 1,
    "not_protonated": not_protonated,
}

# "simple" moves change one replica at a time, "genetic" make a genetic step, "tempering" does exchange same as parallel tempering.
global_change_params = {
    "num_parallel_tempering_tries": 5,
    "num_genetic_tries": 5,
    "prob_dict": {"simple": 0.5, "genetic": 0.25, "tempering": 0.25},
}

# Initial point for all replicas is CC#CC.
init_cg = str2ChemGraph("6#1@1:6@2:6@3:6#1")
# init_cg=str2ChemGraph("6#4")

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

print("Move statistics:")
for k, val in rw.move_statistics().items():
    print(k, ":", val)

# Print analysis of the generated distribution.
print_distribution_analysis(
    rw.histogram,
    betas=betas,
    val_lbound=-0.5,
    val_ubound=len(intervals) - 0.5,
    num_bins=len(intervals),
)
