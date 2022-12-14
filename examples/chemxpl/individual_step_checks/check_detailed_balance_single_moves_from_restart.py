# Randomly generate several molecules and then check that detailed balance is satisfied for each individual step.
import random, sys
import numpy as np
from bmapqml.chemxpl.test_utils import all_procedure_prop_probability_checks
from bmapqml.chemxpl.valence_treatment import str2ChemGraph
from bmapqml.utils import loadpkl

restart_file = sys.argv[1]

if len(sys.argv) > 2:
    seed = int(sys.argv[2])
else:
    seed = 1

random.seed(seed)
np.random.seed(seed)

num_random_couplings = 1

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

possible_elements = ["B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br"]

print("Loading histogram.")
histogram = loadpkl(restart_file)["histogram"]
print("Histogram loaded.")

random.seed(seed)
np.random.seed(seed)

init_tp = random.choice(histogram)

num_mols = 4
num_attempts = 80000  # 40000

randomized_change_params = {
    "possible_elements": possible_elements,
    "nhatoms_range": [1, 9],
    "bond_order_changes": [-1, 1],
    "bond_order_valence_changes": [-2, 2],
    "max_fragment_num": 1,
    "forbidden_bonds": forbidden_bonds,
    "nhatoms_range": nhatoms_range,
}


all_procedure_prop_probability_checks(
    init_tp, num_attempts=num_attempts, print_dicts=True, **randomized_change_params
)
