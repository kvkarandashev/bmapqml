from bmapqml.chemxpl.valence_treatment import ChemGraph
from bmapqml.chemxpl.random_walk import RandomWalk
import random
from bmapqml.chemxpl import ExtGraphCompound
from copy import deepcopy
import math
import numpy as np

random.seed(1)

possible_elements=['P', 'N']

forbidden_bonds=[(15, 15)]

num_MC_steps=100000 #100000

bias_coeff=None
bound_enforcing_coeff=math.log(2.)

randomized_change_params={"max_fragment_num" : 1, "nhatoms_range" : [1, 3], "final_nhatoms_range" : [2, 3],
                        "possible_elements" : possible_elements, "bond_order_changes" : [-1, 1],
                        "forbidden_bonds" : forbidden_bonds}

conserve_stochiometry=False # False True

init_ncharges=[15]
init_bond_orders={}

init_cg=ChemGraph(nuclear_charges=init_ncharges, bond_orders=init_bond_orders, hydrogen_autofill=True)

cur_egc=ExtGraphCompound(chemgraph=init_cg)

histogram=[]
histogram_labels=[]

rw=RandomWalk(bias_coeff=bias_coeff, randomized_change_params=randomized_change_params, conserve_stochiometry=conserve_stochiometry,
                    bound_enforcing_coeff=bound_enforcing_coeff, num_replicas=1)
for MC_step in range(num_MC_steps):
    # If changing randomized_change_params is required mid-simulation they can be updated via *.change_rdkit arguments
    cur_egc=rw.change_molecule(cur_egc, randomized_change_params=randomized_change_params)
    if cur_egc not in histogram_labels:
        histogram_labels.append(deepcopy(cur_egc))
        histogram.append(0)
    cur_egc_id=histogram_labels.index(cur_egc)
    histogram[cur_egc_id]+=1
    print(MC_step, cur_egc)

distr_avs=np.zeros((2,))
distr_avs2=np.zeros((2,))
nvals=np.zeros((2,), dtype=int)
for egc, distr in zip(histogram_labels, histogram):
    print("EGC:", egc)
    print("Distribution:", distr)
    if egc.chemgraph.nhatoms()==1:
        i=0
    else:
        i=1
    distr_avs[i]+=distr
    nvals[i]+=1
    distr_avs2[i]+=distr**2

distr_avs/=nvals
distr_avs2/=nvals

print("Standard deviations:", np.sqrt(distr_avs2-distr_avs**2))
