# A script I used during algorithm testing. Uploaded to showcase how 
# "no_exploration" option works.
from bmapqml.chemxpl.valence_treatment import ChemGraph
from bmapqml.chemxpl.random_walk import RandomWalk, TrajectoryPoint
import random
from bmapqml.chemxpl import ExtGraphCompound
from sortedcontainers import SortedList
from copy import deepcopy
import math

random.seed(1)

possible_elements=['N']

#possible_elements=["F", "Cl"]

forbidden_bonds=None #(15, 15)]

num_MC_steps=100000 #100000

bias_coeff=None
bound_enforcing_coeff=math.log(2.)

randomized_change_params={"max_fragment_num" : 1, "nhatoms_range" : [4, 4], "final_nhatoms_range" : [4, 4],
                        "possible_elements" : possible_elements, "bond_order_changes" : [-1, 1],
                        "forbidden_bonds" : forbidden_bonds}

conserve_stochiometry=False # False True

#init_ncharges=[7, 7, 7]
#init_bond_orders={(0, 1) : 1, (1, 2) : 1}

#init_ncharges=[7, 7, 7, 7]
#init_bond_orders={(0, 1) : 1, (1, 2) : 1, (2, 3) : 1}

init_ncharges=[7, 7, 7, 7]
init_bos=[]
#init_bos.append({(0, 1) : 2, (1, 2) : 1, (2, 3) : 2, (0, 3) : 1})
#init_bos.append({(0, 1) : 1, (1, 2) : 2, (2, 3) : 1, (0, 3) : 1})
#init_bos.append({(0, 1) : 2, (0, 2) : 1, (2, 3) : 2})
#init_bos.append({(0, 1) : 1, (0, 2) : 1, (2, 3) : 2})
#init_bos.append({(0, 1) : 1, (0, 2) : 2, (2, 3) : 1})
init_bos.append({(0, 1) : 1, (1, 2) : 1, (2, 3) : 1, (0, 3) : 1})
#init_bos.append({(0, 1) : 1, (1, 2) : 1, (2, 3) : 1})
init_bos.append({(0, 1) : 1, (1, 2) : 1, (2, 3) : 1, (0, 3) : 1, (1, 3) : 1})
init_bos.append({(0, 1) : 1, (1, 2) : 1, (0, 2) : 1, (0, 3) : 1, (1, 3) : 1, (2, 3) : 1})

init_cgs=[ChemGraph(nuclear_charges=init_ncharges, bond_orders=init_bond_orders, hydrogen_autofill=True) for init_bond_orders in init_bos]

cur_egc=ExtGraphCompound(chemgraph=init_cgs[0])

tps=[TrajectoryPoint(cg=init_cg) for init_cg in init_cgs]

histogram=[]
histogram_labels=[]

rw=RandomWalk(init_egcs=[cur_egc], bias_coeff=bias_coeff, randomized_change_params=randomized_change_params,
                conserve_stochiometry=conserve_stochiometry, bound_enforcing_coeff=bound_enforcing_coeff,
                no_exploration=True, restricted_tps=tps, keep_histogram=False, no_exploration_smove_adjust=True)
for MC_step in range(num_MC_steps):
    # If changing randomized_change_params is required mid-simulation they can be updated via *.change_rdkit arguments
    rw.MC_step_all(randomized_change_params=randomized_change_params)
    cur_egc=rw.cur_tps[0].egc
    if cur_egc not in histogram_labels:
        histogram_labels.append(deepcopy(cur_egc))
        histogram.append(0)
    cur_egc_id=histogram_labels.index(cur_egc)
    histogram[cur_egc_id]+=1
    print(MC_step, cur_egc)

av=.0
av2=.0
nbins=len(histogram)

for egc, distr in zip(histogram_labels, histogram):
    print("EGC:", egc)
    print("Distribution:", distr)
    prob=float(distr)/float(num_MC_steps)
    av+=prob
    av2+=prob**2

av/=nbins
av2/=nbins

stddev=math.sqrt(av2-av**2)

print("prob stddev:", stddev)

#for tp in rw.histogram:
#    print(tp)
#    print(tp.possibilities())
