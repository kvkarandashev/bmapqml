# Do MC optimization for several parallel replicas.
import random
from bmapqml.chemxpl.valence_treatment import ChemGraph
from bmapqml.chemxpl.random_walk import RandomWalk
from bmapqml.chemxpl import ExtGraphCompound
from bmapqml.chemxpl.minimized_functions import sample_local_space
from copy import deepcopy
import numpy as np
from bmapqml.chemxpl import rdkit_descriptors
from bmapqml.chemxpl.utils import xyz_list2mols_extgraph
import argparse
from bmapqml.utils import dump2pkl
from bmapqml.chemxpl.plotting import analyze_random_walk

parser = argparse.ArgumentParser(description='simulation paramters')
parser.add_argument("-Nsteps", default=100000, type=int)
parser.add_argument("-dmin", default=1, type=float)
parser.add_argument("-label", default=1, type=int)
parser.add_argument("-thickness", default=0.25, type=float)
args = parser.parse_args()

np.random.seed(1337+ args.label)
random.seed(1337+args.label)



possible_elements=["C", "O", "N"]

forbidden_bonds=None
ref_beta=8000 #4000.

# None corresponds to greedy optimization, other betas are used in a Metropolis scheme.
betas=[None, None, ref_beta, ref_beta/2, ref_beta/4, ref_beta/8]


num_MC_steps= args.Nsteps
bias_coeff=0.25
vbeta_bias_coeff=1.e-5
bound_enforcing_coeff=1.0




randomized_change_params = {"max_fragment_num": 1, "nhatoms_range": [13,13], "final_nhatoms_range": [13, 13],
                            "possible_elements": possible_elements, "bond_order_changes": [-1, 1],
                            "forbidden_bonds": forbidden_bonds}
global_change_params={"num_parallel_tempering_tries" : 10, "num_genetic_tries" : 10, "prob_dict" : {"simple" : 0.5, "genetic" : 0.25, "tempering" : 0.25}}


output = xyz_list2mols_extgraph(["./asperin.xyz"])
init_cg = output[0].chemgraph
negcs=len(betas)

init_egcs=[ExtGraphCompound(chemgraph=deepcopy(init_cg)) for i in range(negcs)]

fp_type = "MorganFingerprint"
target = "CC(=O)OC1=CC=CC=C1C(=O)O"
X_target = rdkit_descriptors.get_all_FP([target], fp_type)
min_func = sample_local_space(X_target, verbose=True,check_ring=False, pot_type="flat_parabola",fp_type=fp_type , epsilon=5,gamma=args.dmin-args.thickness, sigma=args.dmin)

min_func_name = "chemspacesampler"

current_best=[]


#conserve_stochiometry = True but produces stuff with nitrogen!!!

rw=RandomWalk(bias_coeff=bias_coeff, randomized_change_params=randomized_change_params,
                            bound_enforcing_coeff=bound_enforcing_coeff, betas=betas, min_function=min_func,
                            init_egcs=init_egcs, conserve_stochiometry=True, min_function_name=min_func_name,
                            keep_histogram=True, num_saved_candidates=4, vbeta_bias_coeff=vbeta_bias_coeff, keep_full_trajectory=True)

for MC_step in range(num_MC_steps):
    rw.global_random_change(**global_change_params)
    print("Current best at step:", MC_step)
    for i, cc in enumerate(rw.saved_candidates):
        print(i, cc.func_val, cc.tp.calculated_data["canonical_rdkit"][-1])
    current_best.append(rw.saved_candidates[0].func_val)
    print("Number of moves since last change:", rw.moves_since_changed)


traj=rw.ordered_trajectory()
print("Number of explored molecules:", len(rw.histogram))
dump2pkl(rw.histogram, "histogram.pkl")
dump2pkl(traj, "trajectory.pkl")
ana = analyze_random_walk("histogram.pkl",trajectory="trajectory.pkl", target=target, fp_type = fp_type, model=min_func, name=min_func_name,verbose=True, dmin=args.dmin, thickness=args.thickness)
ana.evaluate_histogram()