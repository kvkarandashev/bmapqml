# Check that in the simplest system of A-B diatomic molecules detailed balance is satisfied.
# Used to verify that the ordering part of the trial genetic step works correctly.
import random
import numpy as np
from bmapqml.chemxpl.random_walk import TrajectoryPoint, randomized_change
from bmapqml.chemxpl.test_utils import check_prop_probability
from bmapqml.chemxpl.valence_treatment import str2ChemGraph
from bmapqml.chemxpl.minimized_functions.toy_problems import Diatomic_barrier

random.seed(1)
np.random.seed(1)

init_cg_str = "9@1:17"

init_cg = str2ChemGraph(init_cg_str)

init_tp = TrajectoryPoint(cg=init_cg)

init_tp_pair = (init_tp, init_tp)

new_pair = tuple([TrajectoryPoint(cg=str2ChemGraph(s)) for s in ["9@1:9", "17@1:17"]])

minimized_function = Diatomic_barrier(possible_nuclear_charges=[9, 17])

ln2 = np.log(2.0)

betas_list = [[ln2, ln2], [ln2, ln2 / 2.0], [ln2 / 2.0, ln2], [None, ln2], [None, None]]

num_attempts = 10000

randomized_change_params = {
    "max_fragment_num": 1,
    "nhatoms_range": [1, 4],
    "final_nhatoms_range": [1, 4],
    "possible_elements": ["F", "Cl"],
    "forbidden_bonds": None,
}

for betas in betas_list:
    print("BETAS:", betas)
    check_prop_probability(
        init_tp_pair,
        new_pair,
        randomized_change_params=randomized_change_params,
        num_attempts=num_attempts,
        min_function=minimized_function,
        betas=betas,
    )
