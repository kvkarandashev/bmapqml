# Do MC optimization for several parallel replicas.

from bmapqml.chemxpl.valence_treatment import ChemGraph
from bmapqml.chemxpl.random_walk import RandomWalk
import random, math
from bmapqml.chemxpl import ExtGraphCompound
from bmapqml.chemxpl.minimized_functions import QM9_properties,multi_obj,Rdkit_properties
from rdkit.Chem.rdmolfiles import MolToSmiles, MolFromSmiles
from bmapqml.chemxpl.utils import xyz2mol_extgraph
from rdkit.Chem.rdmolops import AddHs
from copy import deepcopy
from sortedcontainers import SortedList
from bmapqml.utils import dump2pkl
import pdb
#nuclear_charges, atomic_symbols, xyz_coordinates, add_attr_dict = xyz2mol_extgraph(
#    "./t.xyz")
#pdb.set_trace()
random.seed(1)

possible_elements=['F', "C", "O", "N"] # Are those QM9 elements?

forbidden_bonds=None #[(17, 9)]

#ref_beta=1.
ref_beta=4000.

# None corresponds to greedy optimization, other betas are used in a Metropolis scheme.
betas=[None, None, ref_beta, ref_beta/2, ref_beta/4, ref_beta/8]

num_MC_steps=2000

bias_coeff=.25
vbeta_bias_coeff=1.e-5
bound_enforcing_coeff=1.0

"""
randomized_change_params = {"max_fragment_num": 1, "nhatoms_range": [2, 2], "final_nhatoms_range": [2, 2],
                            "possible_elements": possible_elements, "bond_order_changes": [-1, 1],
                            "forbidden_bonds": forbidden_bonds}
"""



randomized_change_params = {"max_fragment_num": 1, "nhatoms_range": [10, 14], "final_nhatoms_range": [10, 14],
                            "possible_elements": possible_elements, "bond_order_changes": [-1, 1],
                            "forbidden_bonds": forbidden_bonds}
global_change_params={"num_parallel_tempering_tries" : 10, "num_genetic_tries" : 10, "prob_dict" : {"simple" : 0.5, "genetic" : 0.25, "tempering" : 0.25}}


from bmapqml.chemxpl.utils import xyz_list2mols_extgraph

#output = xyz_list2mols_extgraph(["/home/jan/projects/MOLOPT/molopt/examples/chemxpl/MC_minimization/t.xyz"])
output = xyz_list2mols_extgraph(["./t.xyz"])

init_cg = output[0].chemgraph 
#ChemGraph(nuclear_charges=init_ncharges, bond_orders=init_bond_orders, hydrogen_autofill=True)
#pdb.set_trace()
 
#
negcs=len(betas)

init_egcs=[ExtGraphCompound(chemgraph=deepcopy(init_cg)) for i in range(negcs)]

model_path = "/store/common/jan/qm9/"
# "/store/common/jan/"
#min_func = QM9_properties(model_path=model_path, verbose=True)
#min_func_name = "QM9_model"

min_func = Rdkit_properties(model_path)
min_func_name = "max_charge"

current_best=[]

rw=RandomWalk(bias_coeff=bias_coeff, randomized_change_params=randomized_change_params,
                            bound_enforcing_coeff=bound_enforcing_coeff, betas=betas, min_function=min_func,
                            init_egcs=init_egcs, conserve_stochiometry=False, min_function_name="max_charge",
                            keep_histogram=True, num_saved_candidates=4, vbeta_bias_coeff=vbeta_bias_coeff)
for MC_step in range(num_MC_steps):
    rw.global_random_change(**global_change_params)
    # We can access "num_saved_candidates" of best candidates encountered so far as follows:
    print("Current best at step:", MC_step)
    for i, cc in enumerate(rw.saved_candidates):
        print(i, cc.func_val, cc.tp.calculated_data["canonical_rdkit"][-1])
    current_best.append(rw.saved_candidates[0].func_val)
    print("Number of moves since last change:", rw.moves_since_changed)

print("Number of explored molecules:", len(rw.histogram))

dump2pkl(rw.histogram, "QM9_histogram.pkl")
dump2pkl(rw.saved_candidates, "QM9_best_candidates.pkl")
dump2pkl(current_best, "QM9_best_candidate_history.pkl")
