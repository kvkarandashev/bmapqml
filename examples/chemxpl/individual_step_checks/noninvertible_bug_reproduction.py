# Take a "problematic" molecule from the
import random, sys
import numpy as np
from bmapqml.chemxpl.random_walk import TrajectoryPoint
from bmapqml.chemxpl.test_utils import (
    all_procedure_prop_probability_checks,
    check_prop_probability,
)

from bmapqml.chemxpl.valence_treatment import str2ChemGraph

if len(sys.argv) < 2:
    seed = 1
else:
    seed = int(sys.argv[1])

init_cg_str = "16@2@8@9@14:15@3@4@7@14:15@7@12:8@10:8:7#2@13:7#1@12@13:7:6#3:6#3:6#2@11:6#2@14:6#1@13:6#1:6"
trial_cg_str = "16@1@7@8@13:15@6@11:15@3@6@13:8@9:7#2@12:7#1@11@12:7:6#3:6#3:6#2@10:6#2@13:6#1@12:6#1:6"

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

randomized_change_params = {
    "max_fragment_num": 1,
    "nhatoms_range": [1, 15],
    "final_nhatoms_range": [1, 15],
    "possible_elements": possible_elements,
    "bond_order_changes": [-1, 1],
    "forbidden_bonds": forbidden_bonds,
    "not_protonated": not_protonated,
    "added_bond_orders": [1, 2, 3],
    "bond_order_valence_changes": [1, 2],
}


random.seed(seed)
np.random.seed(seed)

num_shuffles = 50

num_attempts = 10000

for _ in range(num_shuffles):
    init_cg = str2ChemGraph(init_cg_str, shuffle=True)

    init_tp = TrajectoryPoint(cg=init_cg)

    trial_cg = str2ChemGraph(init_cg_str, shuffle=True)

    trial_tp = TrajectoryPoint(cg=trial_cg)

    check_prop_probability(
        init_tp, [trial_tp], num_attempts=num_attempts, **randomized_change_params
    )
