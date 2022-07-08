# Do MC optimization for several parallel replicas.

from bmapqml.chemxpl.random_walk import RandomWalk
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

unstable_connection_charges = [7, 8, 9]

forbidden_bonds = []

for i, unstable_connection_charge1 in enumerate(unstable_connection_charges):
    for unstable_connection_charge2 in unstable_connection_charges[i:]:
        forbidden_bonds.append(
            (unstable_connection_charge1, unstable_connection_charge2)
        )

# None corresponds to greedy optimization, other betas are used in a Metropolis scheme.
betas = [None, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0]

make_restart_frequency = 100
num_MC_steps = 1000  # 100000

bias_coeff = None
vbeta_bias_coeff = None

randomized_change_params = {
    "max_fragment_num": 1,
    "nhatoms_range": [4, 9],
    "final_nhatoms_range": [4, 9],
    "possible_elements": possible_elements,
    "bond_order_changes": [-1, 1],
    "forbidden_bonds": forbidden_bonds,
}
global_change_params = {
    "num_parallel_tempering_tries": 5,
    "num_genetic_tries": 5,
    "prob_dict": {"simple": 0.3, "genetic": 0.3, "tempering": 0.3},
}

num_replicas = len(betas)

# We will use QM9's xyz files as starting positions.
mmff_xyz_dir = os.environ["DATA"] + "/QM9_filtered/MMFF_xyzs"
xyz_dir = os.environ["DATA"] + "/QM9_filtered/xyzs"

mmff_xyzs = dirs_xyz_list(os.environ["DATA"] + "/QM9_filtered/MMFF_xyzs")

init_egcs = []

for f in random.sample(mmff_xyzs, num_replicas):
    true_xyz_name = os.path.basename(f)
    qm9_xyz_name = xyz_dir + "/" + true_xyz_name
    SMILES = read_SMILES(qm9_xyz_name)
    print(SMILES)
    init_egcs.append(SMILES_to_egc(SMILES))

min_func = loadpkl(min_func_pkl)

rw = RandomWalk(
    bias_coeff=bias_coeff,
    vbeta_bias_coeff=vbeta_bias_coeff,
    randomized_change_params=randomized_change_params,
    betas=betas,
    min_function=min_func,
    min_function_name=min_func_name,
    init_egcs=init_egcs,
    keep_full_trajectory=True,
    keep_histogram=True,
    make_restart_frequency=make_restart_frequency,
    soft_exit_check_frequency=make_restart_frequency,
    restart_file=restart_file_prefix + str(seed) + ".pkl",
)
for MC_step in range(num_MC_steps):
    # If changing randomized_change_params is required mid-simulation they can be updated via *.change_rdkit arguments
    rw.global_random_change(**global_change_params)

rw.make_restart()
