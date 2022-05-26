# Do MC optimization for several parallel replicas.

from bmapqml.chemxpl.valence_treatment import ChemGraph
from bmapqml.chemxpl.random_walk import RandomWalk
import random, math
from bmapqml.chemxpl import ExtGraphCompound
from bmapqml.chemxpl.minimized_functions import Diatomic_barrier
from copy import deepcopy

random.seed(1)

possible_elements=['Cl', 'F']

forbidden_bonds=None #[(17, 9)]

ln2=math.log(2.)

# None corresponds to greedy optimization, other betas are used in a Metropolis scheme.
betas=[None, ln2, ln2/2]
#betas=[ln2, ln2, ln2, ln2, ln2, ln2]

num_MC_steps=100000 #100000

bias_coeff=None
bound_enforcing_coeff=1.0

randomized_change_params={"max_fragment_num" : 1, "nhatoms_range" : [2, 2], "final_nhatoms_range" : [2, 2],
                        "possible_elements" : possible_elements, "bond_order_changes" : [-1, 1],
                        "forbidden_bonds" : forbidden_bonds}
global_change_params={"num_parallel_tempering_tries" : 5, "num_genetic_tries" : 5, "prob_dict" : {"simple" : 0.5, "genetic" : 0.25, "tempering" : 0.25}}

init_ncharges=[17, 17]
init_bond_orders={(0, 1) : 1}

init_cg=ChemGraph(nuclear_charges=init_ncharges, bond_orders=init_bond_orders, hydrogen_autofill=True)

negcs=len(betas)

init_egcs=[ExtGraphCompound(chemgraph=deepcopy(init_cg)) for i in range(negcs)]

histogram=[[] for i in range(negcs)]
histogram_labels=[[] for i in range(negcs)]

min_func=Diatomic_barrier([9, 17])

rw=RandomWalk(bias_coeff=bias_coeff, randomized_change_params=randomized_change_params,
                            bound_enforcing_coeff=bound_enforcing_coeff, betas=betas, min_function=min_func, init_egcs=init_egcs)
for MC_step in range(num_MC_steps):
    # If changing randomized_change_params is required mid-simulation they can be updated via *.change_rdkit arguments
    rw.global_random_change(**global_change_params)
    cur_egcs=[tp.egc for tp in rw.cur_tps]
    for i in range(negcs):
        cur_egc=cur_egcs[i]
        if cur_egc not in histogram_labels[i]:
            histogram_labels[i].append(deepcopy(cur_egc))
            histogram[i].append(0)
        cur_egc_id=histogram_labels[i].index(cur_egc)
        histogram[i][cur_egc_id]+=1
        print(i, cur_egc)

global_hist_labels=[]
global_histogram=[]

for i, beta in enumerate(betas):
    print("Beta:", beta, "index:", i)
    for egc, distr in zip(histogram_labels[i], histogram[i]):
        print("EGC:", egc)
        print("Distribution:", distr)
    for hl, hv in zip(histogram_labels[i], histogram[i]):
        if hl not in global_hist_labels:
            global_hist_labels.append(hl)
            global_histogram.append(0)
        i=global_hist_labels.index(hl)
        global_histogram[i]+=hv

print("Global histogram:")
for egc, distr in zip(global_hist_labels, global_histogram):
    print(egc, distr)

