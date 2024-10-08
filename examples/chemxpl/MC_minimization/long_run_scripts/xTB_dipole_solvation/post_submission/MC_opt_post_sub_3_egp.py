# Do MC optimization for several parallel replicas.
from bmapqml.chemxpl.random_walk import (
    RandomWalk,
    gen_exp_beta_array,
    TrajectoryPoint,
    InvalidStartingMolecules,
)
from bmapqml.chemxpl.valence_treatment import str2ChemGraph
from bmapqml.chemxpl.ext_graph_compound import ExtGraphCompound
from bmapqml.utils import loadpkl
import random, sys, copy, os
import numpy as np

min_func_pkl = sys.argv[1]

min_func_name = sys.argv[2]

seed = int(sys.argv[3])

bias_strength = sys.argv[4]

valid_starter_strs = sys.argv[5]

restart_file_prefix = "restart_file_"

random.seed(seed)
np.random.seed(seed)

possible_elements = ["B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br"]

not_protonated = [5, 8, 9, 14, 15, 16, 17, 35]
forbidden_bonds = [
    (7, 7),
    (7, 8),
    (8, 8),
    (7, 9),
    (8, 9),
    (9, 9),
    (7, 17),
    (8, 17),
    (9, 17),
    (17, 17),
    (7, 35),
    (8, 35),
    (9, 35),
    (17, 35),
    (35, 35),
    (15, 15),
    (16, 16),
]
nhatoms_range = [1, 15]

# None corresponds to greedy optimization, other betas are used in a Metropolis scheme.
# There are two "greedy" replicas here to verify Guido's comment on checking they don't merge into one.
betas = gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0)

print("Chosen betas:", betas)

# Making a lot of restarts because the bugs are hard to reproduce.
make_restart_frequency = 1000  # 2000
num_MC_steps = 50000  # 50000

bias_coeffs = {"none": None, "weak": 0.2, "stronger": 0.4}
# vbeta_bias_coeffs = {"none": None, "weak": 0.01, "stronger": 0.02}
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

restart_file_name = restart_file_prefix + str(seed) + ".pkl"

all_init_strs = open(valid_starter_strs, "r").readlines()

while True:
    init_egc_val_list = []
    for _ in range(num_replicas):
        new_val = None
        while new_val is None:
            init_str = random.choice(all_init_strs)
            new_egc = ExtGraphCompound(chemgraph=str2ChemGraph(init_str))
            new_tp = TrajectoryPoint(egc=new_egc)
            new_val = min_func(new_tp)
        init_egc_val_list.append((new_val, new_egc))

    init_egc_val_list.sort()

    init_egcs = [egc_val[1] for egc_val in init_egc_val_list]

    try:
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
            restart_file=restart_file_name,
            num_saved_candidates=100,
            delete_temp_data=[],
            histogram_dump_file_prefix="histogram_dump_",
            max_histogram_size=None,
            track_histogram_size=True,
            linear_storage=True,
            greedy_delete_checked_paths=True,
            debug=True,
        )
        print("Successful initialization.")
    except InvalidStartingMolecules:
        print("Failed for initial molecules:")
        print(init_egcs)
        continue
    for i, (beta, tp) in enumerate(zip(betas, rw.cur_tps)):
        print("#INITTP", i, beta, tp, tp.calculated_data[min_func_name])
    break

if os.path.isfile(restart_file_name):
    rw.restart_from()

rw.complete_simulation(num_global_MC_steps=num_MC_steps, **global_change_params)

rw.make_restart()

if hasattr(min_func, "call_counter"):
    print("Number of function calls:", min_func.call_counter)
