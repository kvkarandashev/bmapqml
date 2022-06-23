from bmapqml.chemxpl.valence_treatment import ChemGraph
from bmapqml.chemxpl.random_walk import RandomWalk
import random
from bmapqml.chemxpl import ExtGraphCompound
import math
from bmapqml.chemxpl.modify import add_heavy_atom_chain, remove_heavy_atom, replace_heavy_atom, change_bond_order, change_valence
import numpy as np
import copy

random.seed(1)

possible_elements=['C', 'N']

forbidden_bonds=None

num_MC_steps=10000 #100000

bias_coeff=None
bound_enforcing_coeff=math.log(2.)

# Here you define the probabilities of taking each of the individual smaller moves:
change_prob_dict={add_heavy_atom_chain : .5, remove_heavy_atom : .5, replace_heavy_atom : .25}
# Here you define the probabilities with which the 
global_prob_dict={"simple" : 0.5, "genetic" : 0.25}

randomized_change_params={"max_fragment_num" : 1, "nhatoms_range" : [1, 3], "final_nhatoms_range" : [2, 3],
                        "possible_elements" : possible_elements, "bond_order_changes" : [-1, 1],
                        "forbidden_bonds" : forbidden_bonds, "change_prob_dict" : change_prob_dict}

init_ncharges=[6]
init_bond_orders={}

init_cg=ChemGraph(nuclear_charges=init_ncharges, bond_orders=init_bond_orders, hydrogen_autofill=True)

cur_egc=ExtGraphCompound(chemgraph=init_cg)

histogram=[]
histogram_labels=[]

rw=RandomWalk(init_egcs=[copy.deepcopy(cur_egc) for i in range(3)], bias_coeff=bias_coeff,
            randomized_change_params=randomized_change_params, bound_enforcing_coeff=bound_enforcing_coeff,
                keep_histogram=True)
for MC_step in range(num_MC_steps):
    # If changing randomized_change_params is required mid-simulation they can be updated via *.change_rdkit arguments
    rw.global_random_change(prob_dict=global_prob_dict)
    print(MC_step, rw.cur_tps)

print("Histogram:")
for tp in rw.histogram:
    print(tp, tp.num_visits)

