# Demonstrates that setting add_heavy_atom_chain and remove_heavy_atom probabilities differently does not affect detailed balance.
from bmapqml.chemxpl.valence_treatment import str2ChemGraph
from bmapqml.chemxpl.random_walk import RandomWalk
from bmapqml.chemxpl.modify import add_heavy_atom_chain, remove_heavy_atom
import random
from bmapqml.chemxpl import ExtGraphCompound
import math

random.seed(1)

possible_elements = ["C"]

forbidden_bonds = None

num_MC_steps = 100000

bias_coeff = None

bound_enforcing_coeff = math.log(2.0)

randomized_change_params = {
    "max_fragment_num": 1,
    "nhatoms_range": [1, 4],
    "final_nhatoms_range": [2, 3],
    "possible_elements": possible_elements,
    "bond_order_changes": [-1, 1],
    "forbidden_bonds": forbidden_bonds,
    "change_prob_dict": {
        add_heavy_atom_chain: 0.25,
        remove_heavy_atom: 0.75,
    },
}

init_str = "6#4"
num_replicas = 1

rw = RandomWalk(
    init_egcs=[
        ExtGraphCompound(chemgraph=str2ChemGraph(init_str)) for _ in range(num_replicas)
    ],
    bias_coeff=bias_coeff,
    randomized_change_params=randomized_change_params,
    visit_num_count_acceptance=True,
    bound_enforcing_coeff=bound_enforcing_coeff,
    num_replicas=num_replicas,
    keep_histogram=True,
)
for MC_step in range(num_MC_steps):
    # If changing randomized_change_params is required mid-simulation they can be updated via *.change_rdkit arguments
    rw.MC_step_all()
    print(MC_step, rw.cur_tps)

for tp in rw.histogram:
    print(tp, tp.num_visits)
