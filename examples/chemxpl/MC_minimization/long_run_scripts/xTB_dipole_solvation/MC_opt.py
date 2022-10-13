# Do MC optimization for several parallel replicas.
from bmapqml.chemxpl.random_walk import RandomWalk, gen_beta_array
from bmapqml.chemxpl.utils import SMILES_to_egc
from bmapqml.utils import loadpkl
import random, sys, copy
import numpy as np

min_func_pkl = sys.argv[1]

min_func_name = sys.argv[2]

seed = int(sys.argv[3])

bias_strength = sys.argv[4]

restart_file_prefix = "restart_file_"

random.seed(seed)
np.random.seed(seed)

init_SMILES = "C"

init_egc = SMILES_to_egc(init_SMILES)

possible_elements = ["C", "N", "O", "F"]

# forbidden_bonds = [(7, 9), (8, 9), (9, 9)]
forbidden_bonds = [(7, 7), (7, 8), (8, 8), (7, 9), (8, 9), (9, 9)]

# None corresponds to greedy optimization, other betas are used in a Metropolis scheme.
# There are two "greedy" replicas here to verify Guido's comment on checking they don't merge into one.
betas = gen_beta_array(4, 8.0, 0.5, 4.0, 0.25)

print("Chosen betas:", betas)

make_restart_frequency = 2000
num_MC_steps = 50000  # 50000

bias_coeffs = {"none": None, "weak": 0.2, "stronger": 0.4}
vbeta_bias_coeffs = {"none": None, "weak": 0.01, "stronger": 0.02}


# bias_coeff = None
# vbeta_bias_coeff = None
bias_coeff = bias_coeffs[bias_strength]
vbeta_bias_coeff = bias_coeffs[bias_strength]


randomized_change_params = {
    "max_fragment_num": 1,
    "nhatoms_range": [1, 9],
    "final_nhatoms_range": [1, 9],
    "possible_elements": possible_elements,
    "bond_order_changes": [-1, 1],
    "forbidden_bonds": forbidden_bonds,
}
global_change_params = {
    "num_parallel_tempering_tries": 32,
    "num_genetic_tries": 8,
    "prob_dict": {"simple": 0.3, "genetic": 0.3, "tempering": 0.3},
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
)

for MC_step in range(num_MC_steps):
    # If changing randomized_change_params is required mid-simulation they can be updated via *.change_rdkit arguments
    rw.global_random_change(**global_change_params)

rw.make_restart()

if hasattr(min_func, "call_counter"):
    print("Number of function calls:", min_func.call_counter)
