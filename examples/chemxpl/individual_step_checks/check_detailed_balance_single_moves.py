# Randomly generate several molecules and then check that detailed balance is satisfied for each individual step.
import random
import numpy as np
from bmapqml.chemxpl.random_walk import TrajectoryPoint, randomized_change
from bmapqml.chemxpl.test_utils import check_prop_probability
from bmapqml.chemxpl.valence_treatment import str2ChemGraph

random.seed(1)
np.random.seed(1)

init_cg_str = "6#3@1:15#1@2:6#3"

init_cg = str2ChemGraph(init_cg_str)

init_tp = TrajectoryPoint(cg=init_cg)

num_mols = 4
num_attempts = 100

randomized_change_params = {
    "possible_elements": ["C", "P"],
    "nhatoms_range": [1, 9],
    "bond_order_changes": [-1, 1],
    "bond_order_valence_changes": [-2, 2],
}

neighbors = []
while len(neighbors) != num_mols:
    new_tp, _ = randomized_change(init_tp, **randomized_change_params)
    if new_tp not in neighbors:
        neighbors.append(new_tp)

check_prop_probability(init_tp, neighbors, **randomized_change_params)
