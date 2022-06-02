# A version of detailed_balance_resonance_structures.py that SHOULD save some time by
# saving all the graph-related data in the histogram.

from bmapqml.chemxpl.valence_treatment import ChemGraph
from bmapqml.chemxpl.random_walk import RandomWalk
import random
from bmapqml.chemxpl import ExtGraphCompound
from copy import deepcopy
import math

random.seed(1)

possible_elements=['N']

forbidden_bonds=None #(15, 15)]

num_MC_steps=200000 #200000

bias_coeff=None
bound_enforcing_coeff=math.log(2.)

randomized_change_params={"max_fragment_num" : 1, "nhatoms_range" : [3, 4], "final_nhatoms_range" : [4, 4],
                        "possible_elements" : possible_elements, "bond_order_changes" : [-1, 1],
                        "forbidden_bonds" : forbidden_bonds}

conserve_stochiometry=False # False True

#init_ncharges=[7, 7, 7]
#init_bond_orders={(0, 1) : 1, (1, 2) : 1}

init_ncharges=[7, 7, 7, 7]
init_bond_orders={(0, 1) : 1, (1, 2) : 1, (2, 3) : 1}


init_cg=ChemGraph(nuclear_charges=init_ncharges, bond_orders=init_bond_orders, hydrogen_autofill=True)

cur_egc=ExtGraphCompound(chemgraph=init_cg)

histogram=[]
histogram_labels=[]

rw=RandomWalk(bias_coeff=bias_coeff, randomized_change_params=randomized_change_params, conserve_stochiometry=conserve_stochiometry,
                        bound_enforcing_coeff=bound_enforcing_coeff, keep_histogram=True, init_egcs=[cur_egc])
for MC_step in range(num_MC_steps):
    # If changing randomized_change_params is required mid-simulation they can be updated via *.change_rdkit arguments
    rw.MC_step_all()
    print(MC_step, rw.cur_tps)

for tp in rw.histogram:
    print("EGC:", tp)
    print("Distribution:", tp.num_visits)
