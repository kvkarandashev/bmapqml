# The script verifies that Monte Carlo trajectories generated with RandomWalk objects follow the correct distribution when no bias is applied.
from bmapqml.chemxpl.valence_treatment import str2ChemGraph
from bmapqml.chemxpl.random_walk import RandomWalk
import random
import numpy as np
from bmapqml.chemxpl import ExtGraphCompound
from bmapqml.chemxpl.utils import print_distribution_analysis
from bmapqml.chemxpl.minimized_functions import Diatomic_barrier
from copy import deepcopy

# from sortedcontainers import SortedDict

random.seed(1)
np.random.seed(1)

# Constraints on chemical space - molecules with only C and P heavy atoms, maximum 4 heavy atoms large,
# bonds between two phosphorus atoms are forbidden.
possible_elements = ["F", "Cl"]

forbidden_bonds = None  # [(15, 15)]

min_func = Diatomic_barrier([9, 17])

nhatoms_range = [2, 2]

ln2 = np.log(2.0)

betas = [None, 2.0 * ln2, ln2, ln2 / 2.0, ln2 / 4.0]

num_MC_steps = 4000  # 100000


randomized_change_params = {
    "max_fragment_num": 1,
    "nhatoms_range": nhatoms_range,
    "final_nhatoms_range": nhatoms_range,
    "possible_elements": possible_elements,
    "bond_order_changes": [-1, 1],
    "forbidden_bonds": forbidden_bonds,
}

# "simple" moves change one replica at a time, "genetic" make a genetic step, "tempering" does exchange same as parallel tempering.
global_change_params = {
    "num_parallel_tempering_tries": 32,
    "num_genetic_tries": 32,
    "prob_dict": {"simple": 0.5, "genetic": 0.25, "tempering": 0.25},
}

# Initial point for all replicas is CC#CC.
init_cg = str2ChemGraph("17@1:17")
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

print_distribution_analysis(
    rw.histogram, betas, val_lbound=-0.5, val_ubound=2.5, num_bins=3
)

# The following is a short analysis for easy verification that the distribution is the correct one.

# for beta_id, beta in enumerate(betas):
#    print("\n\nBeta:", beta)
#    distr=SortedDict()
# Cycle over all graphs visited during the simulation.
#    for cur_tp in rw.histogram:
#        cur_num_visits = cur_tp.visit_num(
#            beta_id
#        )  # visit_num(beta_id) gives number of times graph was visited by replica with id beta_id.
#        val = min_func(cur_tp)
#        distr[val]=cur_num_visits

#    for val, d in distr.items():
#        print("Function value:", val, "distribution:", d)
#    if beta is not None:
#        tot_distr=sum(list(distr.values()))
#        boltz_weights=[np.exp(-beta*val) for val in distr.keys()]
#        boltz_weights/=sum(boltz_weights)
#        print("Pot-hist deviation:")
#        for bw, (val, d) in zip(boltz_weights, distr.items()):
#            print(val, np.log(bw*tot_distr/d))
