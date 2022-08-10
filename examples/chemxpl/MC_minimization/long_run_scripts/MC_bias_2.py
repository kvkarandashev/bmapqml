# Do MC optimization for several parallel replicas.

from bmapqml.chemxpl.random_walk import RandomWalk, InvalidStartingMolecules
from bmapqml.chemxpl.utils import SMILES_to_egc
from bmapqml.dataset_processing.qm9_format_specs import read_SMILES
from bmapqml.test_utils import dirs_xyz_list
from bmapqml.utils import loadpkl
import random, sys, os
import numpy as np

min_func_pkl = sys.argv[1]

min_func_name = sys.argv[2]

seed = int(sys.argv[3])

# restart_file_prefix=sys.argv[4]

restart_file_prefix = "restart_file_"

random.seed(seed)
np.random.seed(seed)

possible_elements = ["C", "N", "O", "F"]

# forbidden_bonds = [(7, 9), (8, 9), (9, 9)]
forbidden_bonds = [(7, 7), (7, 8), (8, 8), (7, 9), (8, 9), (9, 9)]

# None corresponds to greedy optimization, other betas are used in a Metropolis scheme.
betas = [None, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5]

make_restart_frequency = 2000
num_MC_steps = 500000

bias_coeff = 0.2
vbeta_bias_coeff = 0.01

randomized_change_params = {
    "max_fragment_num": 1,
    "nhatoms_range": [4, 9],
    "final_nhatoms_range": [4, 9],
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

# We will use QM9's xyz files as starting positions.
mmff_xyz_dir = os.environ["DATA"] + "/QM9_filtered/MMFF_xyzs"
xyz_dir = os.environ["DATA"] + "/QM9_filtered/xyzs"

mmff_xyzs = dirs_xyz_list(os.environ["DATA"] + "/QM9_filtered/MMFF_xyzs")

if min_func_pkl == "TRIAL_RUN":
    from bmapqml.chemxpl.minimized_functions import ChargeSum

    min_func = ChargeSum()
else:
    min_func = loadpkl(min_func_pkl)

rw = RandomWalk(
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
    delete_temp_data=["coord_info", "res_dict"],
    histogram_dump_file_prefix="histogram_dump_",
    max_histogram_size=None,
    track_histogram_size=True,
)

max_num_attempts = 1000
num_attempts = 0

while True:
    print("Attempting to initialize:")
    init_egcs = []
    for f in random.sample(mmff_xyzs, num_replicas):
        true_xyz_name = os.path.basename(f)
        qm9_xyz_name = xyz_dir + "/" + true_xyz_name
        SMILES = read_SMILES(qm9_xyz_name)
        print(SMILES)
        init_egcs.append(SMILES_to_egc(SMILES))
    try:
        rw.init_cur_tps(init_egcs=init_egcs)
    except InvalidStartingMolecules:
        num_attempts += 1
        print("Initial set invalid.")
        if num_attempts == max_num_attempts:
            quit()
        continue
    break

for MC_step in range(num_MC_steps):
    # If changing randomized_change_params is required mid-simulation they can be updated via *.change_rdkit arguments
    rw.global_random_change(**global_change_params)

rw.make_restart()

if hasattr(min_func, "call_counter"):
    print("Number of function calls:", min_func.call_counter)
