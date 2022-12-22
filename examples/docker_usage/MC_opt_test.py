# Do MC optimization for several parallel replicas.
from bmapqml.chemxpl.random_walk import RandomWalk, gen_exp_beta_array
from bmapqml.chemxpl.utils import SMILES_to_egc
from bmapqml.utils import loadpkl
import random, sys, copy
import numpy as np

min_func_pkl = sys.argv[1]  # use something from examples/chemxpl/minfunc_creation

min_func_name = "min_func"  # sys.argv[2]

seed = 1  # int(sys.argv[3])

bias_strength = "none"  # sys.argv[4]

restart_file_prefix = "restart_file_"

random.seed(seed)
np.random.seed(seed)

init_SMILES = "C"

init_egc = SMILES_to_egc(init_SMILES)

possible_elements = ["C", "N", "O", "F"]

# forbidden_bonds = [(7, 9), (8, 9), (9, 9)]
forbidden_bonds = [(7, 7), (7, 8), (8, 8), (7, 9), (8, 9), (9, 9)]
not_protonated = [8, 9]
nhatoms_range = [1, 9]

# None corresponds to greedy optimization, other betas are used in a Metropolis scheme.
# There are two "greedy" replicas here to verify Guido's comment on checking they don't merge into one.
betas = gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0)

print("Chosen betas:", betas)

make_restart_frequency = 200
num_MC_steps = 600  # 50000

bias_coeffs = {"none": None, "weak": 0.2, "stronger": 0.4}
# vbeta_bias_coeffs = {"none": None, "weak": 0.01, "stronger": 0.02}
# Disabled biasing of greedy replicas in order they systematically checked all minimization options instead.
vbeta_bias_coeffs = {"none": None, "weak": None, "stronger": None}


# bias_coeff = None
# vbeta_bias_coeff = None
bias_coeff = bias_coeffs[bias_strength]
vbeta_bias_coeff = bias_coeffs[bias_strength]


randomized_change_params = {
    "max_fragment_num": 1,
    "nhatoms_range": nhatoms_range,
    "final_nhatoms_range": nhatoms_range,
    "possible_elements": possible_elements,
    "bond_order_changes": [-1, 1],
    "forbidden_bonds": forbidden_bonds,
    "not_protonated": not_protonated,
    "added_bond_orders": [1, 2, 3],
}
global_change_params = {
    "num_parallel_tempering_tries": 128,
    "num_genetic_tries": 32,
    "prob_dict": {"simple": 0.6, "genetic": 0.2, "tempering": 0.2},
}

num_replicas = len(betas)

if min_func_pkl == "TRIAL_RUN":
    from bmapqml.chemxpl.minimized_functions import ChargeSum

    min_func = ChargeSum()
else:
    min_func = loadpkl(min_func_pkl)

init_egcs = [copy.deepcopy(init_egc) for _ in range(num_replicas)]

rw = RandomWalk(
    init_egcs=init_egcs,
    bias_coeff=bias_coeff,
    vbeta_bias_coeff=vbeta_bias_coeff,
    randomized_change_params=randomized_change_params,
    betas=betas,
    min_function=min_func,
    min_function_name=min_func_name,
    keep_histogram=True,
    keep_full_trajectory=True,
    make_restart_frequency=make_restart_frequency,
    soft_exit_check_frequency=make_restart_frequency,
    restart_file=restart_file_prefix + str(seed) + ".pkl",
    num_saved_candidates=100,
    delete_temp_data=[],
    histogram_dump_file_prefix="histogram_dump_",
    max_histogram_size=None,
    track_histogram_size=True,
    linear_storage=True,
    greedy_delete_checked_paths=True,
    debug=True,
)

for MC_step in range(num_MC_steps):
    # If changing randomized_change_params is required mid-simulation they can be updated via *.change_rdkit arguments
    rw.global_random_change(**global_change_params)

rw.make_restart()

if hasattr(min_func, "call_counter"):
    print("Number of function calls:", min_func.call_counter)
