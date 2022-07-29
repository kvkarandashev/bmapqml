from bmapqml.chemxpl.utils import all_egc_from_tar
from bmapqml.chemxpl.valence_treatment import ChemGraph
from bmapqml.chemxpl.random_walk import RandomWalk
import random, math
from bmapqml.chemxpl import ExtGraphCompound
from copy import deepcopy
import numpy as np

random.seed(1)

possible_elements = ["P", "N"]

forbidden_bonds = [(15, 15)]
# forbidden_bonds=None

MC_step_num = 50000  # 50000

bias_coeff = None

randomized_change_params = {
    "max_fragment_num": 1,
    "nhatoms_range": [1, 3],
    "final_nhatoms_range": [2, 3],
    "possible_elements": possible_elements,
    "bond_order_changes": [-1, 1],
    "forbidden_bonds": forbidden_bonds,
    "cross_coupling_fragment_ratio_range": [0.1, 0.7],
}

conserve_stochiometry = False  # False True

init_ncharges = [15, 7]
init_bond_orders = {(0, 1): 1}

init_cg = ChemGraph(
    nuclear_charges=init_ncharges, bond_orders=init_bond_orders, hydrogen_autofill=True
)

init_egc = ExtGraphCompound(chemgraph=init_cg)

histogram_labels = []
histogram = []

num_egcs = 8

egc_list = [deepcopy(init_egc) for egc_counter in range(num_egcs)]

num_gen_moves = 5

rw = RandomWalk(
    randomized_change_params=randomized_change_params,
    conserve_stochiometry=conserve_stochiometry,
    num_replicas=num_egcs,
    bound_enforcing_coeff=np.log(2.0),
)
for MC_step in range(MC_step_num):
    # If changing randomized_change_params is required mid-simulation they can be updated via *.change_rdkit arguments
    egc_list = rw.change_molecule_list(
        egc_list, randomized_change_params=randomized_change_params
    )
    egc_list = rw.genetic_change_molecule_list(
        egc_list,
        num_attempted_changes=num_gen_moves,
        randomized_change_params=randomized_change_params,
    )
    for cur_egc in egc_list:
        if cur_egc not in histogram_labels:
            histogram_labels.append(deepcopy(cur_egc))
            histogram.append(0)
        cur_id = histogram_labels.index(cur_egc)
        histogram[cur_id] += 1
    print("MC step:", MC_step)

print(
    "Genetic step statistics:",
    rw.num_accepted_cross_couplings,
    rw.num_attempted_cross_couplings,
)

nhatoms_lists = [[1], [2, 3]]

for nhatoms_list in nhatoms_lists:
    distrs = []
    for egc, distr in zip(histogram_labels, histogram):
        if egc.chemgraph.nhatoms() in nhatoms_list:
            print("EGC:", egc)
            print("Distribution:", distr)
            distrs.append(distr)

    distrs = np.array(distrs)
    print("nhatoms available:", nhatoms_list)
    print("Average:", np.mean(distrs))
    print("Standard deviation:", np.std(distrs))
