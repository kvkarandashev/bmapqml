import sys

sys.path.append("../../../")
sys.path.append("../../../../Graph2Structure")
from bmapqml.chemxpl.random_walk import RandomWalk, TrajectoryPoint
from bmapqml.chemxpl.utils import SMILES_to_egc
import random
import math
from bmapqml.chemxpl.modify import (
    add_heavy_atom_chain,
    remove_heavy_atom,
    replace_heavy_atom,
    change_bond_order,
    change_valence,
)

random.seed(4)

# The function for finding the distance between to EGC files.
def EGC_walking_distance(egc1, egc2, rw, max_num_MC_steps=10000):

    rw.clear_histogram_visit_data()
    tp_start = TrajectoryPoint(egc=egc1)
    tp_final = TrajectoryPoint(egc=egc2)

    rw.cur_tps = [tp_start]
    for MC_step in range(max_num_MC_steps):
        if tp_final in rw.cur_tps:
            return MC_step
        rw.MC_step_all()
    return None


# Distance, but we start from SMILES.
def SMILES_walking_distance(SMILES1, SMILES2, *args, **kwargs):

    egc1 = SMILES_to_egc(SMILES1)  # , egc_hydrogen_autofill=True)
    egc2 = SMILES_to_egc(SMILES2)  # , egc_hydrogen_autofill=True)

    return EGC_walking_distance(egc1, egc2, *args, **kwargs)


# Parameters of the random walk.
possible_elements = ["C", "P"]

forbidden_bonds = None

max_num_MC_steps = 10000  # 100000

bias_coeff = None
bound_enforcing_coeff = math.log(2.0)

# Here you define the probabilities of taking each of the individual smaller moves:
# If we define it as a list the probabilities are equal:
change_prob_dict = [
    add_heavy_atom_chain,
    remove_heavy_atom,
    replace_heavy_atom,
    change_bond_order,
    change_valence,
]

# Note that using this dictionary would make SMILES2 inaccessible from SMILES1
# change_prob_dict={add_heavy_atom_chain : .5, remove_heavy_atom : .5}

randomized_change_params = {
    "max_fragment_num": 1,
    "nhatoms_range": [1, 3],
    "final_nhatoms_range": [2, 3],
    "possible_elements": possible_elements,
    "bond_order_changes": [-1, 1],
    "forbidden_bonds": forbidden_bonds,
    "change_prob_dict": change_prob_dict,
}

rw = RandomWalk(
    bias_coeff=bias_coeff,
    randomized_change_params=randomized_change_params,
    bound_enforcing_coeff=bound_enforcing_coeff,
    keep_histogram=True,
    num_replicas=1,
)

SMILES1 = "C"
SMILES2 = "C=P=C"

num_attempts = 10

for attempt_counter in range(num_attempts):
    dist = SMILES_walking_distance(
        SMILES1, SMILES2, rw, max_num_MC_steps=max_num_MC_steps
    )
    print(dist)

# Also check how many molecules were encountered by rw.
print("Number of molecules encountered by rw:", len(rw.histogram))
