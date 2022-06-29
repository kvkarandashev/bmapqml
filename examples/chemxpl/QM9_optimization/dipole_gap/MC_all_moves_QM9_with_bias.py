# Do MC optimization for several parallel replicas.

from bmapqml.chemxpl.random_walk import RandomWalk
import random, math
from bmapqml.chemxpl import ExtGraphCompound
from bmapqml.chemxpl.minimized_functions import QM9_properties,multi_obj
from copy import deepcopy
from bmapqml.utils import dump2pkl
import numpy as np

random.seed(1337)
possible_elements=['F', "C", "O", "N"]
forbidden_bonds= [(8,8), (7,8), (8,9), (7,9), (7,7) ]
ref_beta=4000.
# None corresponds to greedy optimization, other betas are used in a Metropolis scheme.
betas=[None, None, ref_beta, ref_beta/2, ref_beta/4, ref_beta/8]
num_MC_steps=100000
bias_coeff=.25
vbeta_bias_coeff=1.e-5
bound_enforcing_coeff=1.0

randomized_change_params = {"max_fragment_num": 1, "nhatoms_range": [9,9], "final_nhatoms_range": [9, 9],
                            "possible_elements": possible_elements, "bond_order_changes": [-1, 1],
                            "forbidden_bonds": forbidden_bonds}
global_change_params={"num_parallel_tempering_tries" : 10, "num_genetic_tries" : 10, "prob_dict" : {"simple" : 0.5, "genetic" : 0.25, "tempering" : 0.25}}


from bmapqml.chemxpl.utils import xyz_list2mols_extgraph
output = xyz_list2mols_extgraph(["./t.xyz"])

init_cg = output[0].chemgraph 
negcs=len(betas)
init_egcs=[ExtGraphCompound(chemgraph=deepcopy(init_cg)) for i in range(negcs)]

"""
Average values of the properties in QM9
    6.8  eV for band gap
    -1.9  eV for atomization energy
    2.67  debye for dipole moment
"""


WEIGHTS = np.array([ (1/2.67), (1/6.8)])
ML_PATH = "/store/common/jan/qm9_removed/ml/"
min_func = multi_obj(
    [QM9_properties(ML_PATH+"KRR_102000_mu",verbose=False, max=True),QM9_properties(ML_PATH+"KRR_102000_gap",verbose=False, max=True)
], WEIGHTS,max=True, verbose=True)

min_func_name = "multi_obj"

current_best=[]

rw=RandomWalk(bias_coeff=bias_coeff, randomized_change_params=randomized_change_params,
                            bound_enforcing_coeff=bound_enforcing_coeff, betas=betas, min_function=min_func,
                            init_egcs=init_egcs, conserve_stochiometry=False, min_function_name=min_func_name,
                            keep_histogram=True, num_saved_candidates=4, vbeta_bias_coeff=vbeta_bias_coeff,keep_full_trajectory=True)
for MC_step in range(num_MC_steps):
    rw.global_random_change(**global_change_params)
    print("Current best at step:", MC_step)
    for i, cc in enumerate(rw.saved_candidates):
        print(i, cc.func_val, cc.tp.calculated_data["canonical_rdkit"][-1], cc.tp.first_MC_step_encounter, cc.tp.first_global_MC_step_encounter)
    current_best.append(rw.saved_candidates[0].func_val)
    print("Number of moves since last change:", rw.moves_since_changed)

print("Number of explored molecules:", len(rw.histogram))

dump2pkl(rw.histogram, "QM9_histogram.pkl")
dump2pkl(rw.saved_candidates, "QM9_best_candidates.pkl")
dump2pkl(current_best, "QM9_best_candidate_history.pkl")


traj=rw.ordered_trajectory()
dump2pkl(traj, "QM9_trajectory.pkl")
