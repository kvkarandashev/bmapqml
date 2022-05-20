from molopt.chemxpl.utils import all_egc_from_tar
from molopt.chemxpl.valence_treatment import ChemGraph
from molopt.chemxpl.random_walk import RandomWalk
import random, math
from chemxpl import ExtGraphCompound
from copy import deepcopy
import numpy as np

random.seed(1)

possible_elements=['P', 'N']

forbidden_bonds=[(15, 15)]
#forbidden_bonds=None

init_gen_MC_steps=1000

detailed_balance_MC_steps=4000

bias_coeff=None

randomized_change_params={"max_fragment_num" : 1, "nhatoms_range" : [2, 3], "final_nhatoms_range" : [2, 3],
                        "possible_elements" : possible_elements, "bond_order_changes" : [-1, 1],
                        "forbidden_bonds" : forbidden_bonds, "cross_coupling_fragment_ratio_range" : [0.1, 0.7]}

conserve_stochiometry=False # False True

init_ncharges=[15, 7]
init_bond_orders={(0, 1) : 1}

init_cg=ChemGraph(nuclear_charges=init_ncharges, bond_orders=init_bond_orders, hydrogen_autofill=True)

init_egc=ExtGraphCompound(chemgraph=init_cg)

histogram_labels=[]

num_egcs=8

egc_list=[deepcopy(init_egc) for egc_counter in range(num_egcs)]

num_gen_moves=5

rw=RandomWalk(randomized_change_params=randomized_change_params, conserve_stochiometry=conserve_stochiometry)
for MC_step in range(init_gen_MC_steps):
    # If changing randomized_change_params is required mid-simulation they can be updated via *.change_rdkit arguments
    egc_list=rw.change_molecule_list(egc_list, randomized_change_params=randomized_change_params)
    for cur_egc in egc_list:
        if cur_egc not in histogram_labels:
            histogram_labels.append(deepcopy(cur_egc))
    print("Initial generation MC step:", MC_step)

histogram=np.zeros(len(histogram_labels), dtype=int)

new_egc_list=deepcopy(histogram_labels)

for MC_step in range(detailed_balance_MC_steps):
    new_egc_list=rw.genetic_change_molecule_list(histogram_labels, num_attempted_changes=num_gen_moves, randomized_change_params=randomized_change_params)
    for cur_egc in new_egc_list:
        if cur_egc not in histogram_labels:
            print("Increase init_gen_MC_steps?")
            quit()
        hist_id=histogram_labels.index(cur_egc)
        histogram[hist_id]+=1
    print("Genetic check MC step:", MC_step)

print("Genetic step statistics:", rw.num_accepted_cross_couplings, rw.num_attempted_cross_couplings)

for egc, distr in zip(histogram_labels, histogram):
    print("EGC:", egc)
    print("Distribution:", distr)
