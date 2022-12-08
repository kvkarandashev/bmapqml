# Check that probability balance is well calculated for larger molecules.
import random
import numpy as np
from bmapqml.chemxpl.random_walk import TrajectoryPoint, randomized_change
from bmapqml.chemxpl.test_utils import check_prop_probability
from bmapqml.chemxpl.modify import randomized_cross_coupling
from bmapqml.chemxpl.valence_treatment import str2ChemGraph
from bmapqml.chemxpl.minimized_functions.toy_problems import Diatomic_barrier

random.seed(1)
np.random.seed(1)

chemgraph_strings = [
    "6#3@1:6@2:7@3:6#1@4:6@1@5:9",
    "9@1:6@2@3:6#1@4:6#1@5:6#1@5:7@6:6#2@7:6#3",
]
cgs = [str2ChemGraph(cg_str) for cg_str in chemgraph_strings]

tps = tuple([TrajectoryPoint(cg=cg) for cg in cgs])

nhatoms_range = [1, 9]

num_new_pairs = 4

new_pairs = []

attempts_to_generate = 40000

for _ in range(attempts_to_generate):
    new_cg_pair, prob_bal = randomized_cross_coupling(cgs, nhatoms_range=nhatoms_range)
    if new_cg_pair is None:
        continue
    tnew_pair = tuple([TrajectoryPoint(cg=cg) for cg in new_cg_pair])
    if tnew_pair not in new_pairs:
        new_pairs.append(tnew_pair)
        if len(new_pairs) == num_new_pairs:
            break


minimized_function = Diatomic_barrier(possible_nuclear_charges=[9, 17])

ln2 = np.log(2.0)

# betas = [ln2, ln2 / 2.0]
betas = [ln2, ln2]

num_attempts = 10000

randomized_change_params = {
    "nhatoms_range": nhatoms_range,
}

print("BETAS:", betas)
check_prop_probability(
    tps,
    new_pairs,
    randomized_change_params=randomized_change_params,
    num_attempts=num_attempts,
    min_function=minimized_function,
    betas=betas,
)
