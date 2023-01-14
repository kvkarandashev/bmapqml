# Randomly generate several molecules and then check that detailed balance is satisfied for each individual step.
import random, sys
import numpy as np
from bmapqml.chemxpl.random_walk import TrajectoryPoint
from bmapqml.chemxpl.test_utils import all_procedure_prop_probability_checks
from bmapqml.chemxpl.valence_treatment import str2ChemGraph

seed = 1

# Good examples of init_cg_str: 6#3@1:16@2:15, 6#3@1:16@2:15@3@4:16#1:16#1, 16#1@1:16@2:15
# init_cg_str difficult for change_valence function in modify.py: 15#2@1@2:15#1@3:15@3:6 15#3@1@3:15#2@2:15@3:6 16#3@1@2@3:16@2@3:6#1:6

possible_elements = None
if len(sys.argv) < 2:
    init_cg_str = "6#3@1:15#1@2:6#3"
    possible_elements = ["C", "P"]
else:
    init_cg_str = sys.argv[1]
    if len(sys.argv) > 2:
        seed = int(sys.argv[2])

random.seed(seed)
np.random.seed(seed)

init_cg = str2ChemGraph(init_cg_str, shuffle=True)

if possible_elements is None:
    possible_elements = []
    for hatom in init_cg.hatoms:
        el = hatom.element_name()
        if el not in possible_elements:
            possible_elements.append(el)

init_tp = TrajectoryPoint(cg=init_cg)

num_mols = 4
num_attempts = 40000

randomized_change_params = {
    "possible_elements": possible_elements,
    "nhatoms_range": [1, 9],
    "bond_order_changes": [-1, 1],
    "bond_order_valence_changes": [-2, 2],
    "max_fragment_num": 1,
}


all_procedure_prop_probability_checks(
    init_tp, num_attempts=num_attempts, print_dicts=True, **randomized_change_params
)
