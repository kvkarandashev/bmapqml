# Do MC optimization for several parallel replicas.

from bmapqml.chemxpl.valence_treatment import ChemGraph
from bmapqml.chemxpl.random_walk import RandomWalk
import random, math
from bmapqml.chemxpl import ExtGraphCompound
from bmapqml.chemxpl.minimized_functions import QM9_properties,multi_obj
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

possible_elements=['Cl', 'F', "C", "O", "N"]

forbidden_bonds=None #[(17, 9)]

ref_beta=1.
#ref_beta=2000.

# None corresponds to greedy optimization, other betas are used in a Metropolis scheme.
betas=[None, ref_beta, ref_beta/2]

num_MC_steps=100

bias_coeff=None
bound_enforcing_coeff=1.0

"""
randomized_change_params = {"max_fragment_num": 1, "nhatoms_range": [2, 2], "final_nhatoms_range": [2, 2],
                            "possible_elements": possible_elements, "bond_order_changes": [-1, 1],
                            "forbidden_bonds": forbidden_bonds}
"""



randomized_change_params = {"max_fragment_num": 1, "nhatoms_range": [13, 13], "final_nhatoms_range": [13, 13],
                            "possible_elements": possible_elements, "bond_order_changes": [-1, 1],
                            "forbidden_bonds": forbidden_bonds}
global_change_params={"num_parallel_tempering_tries" : 5, "num_genetic_tries" : 5, "prob_dict" : {"simple" : 0.5, "genetic" : 0.25, "tempering" : 0.25}}


from bmapqml.chemxpl.utils import xyz_list2mols_extgraph

#output = xyz_list2mols_extgraph(["/home/jan/projects/MOLOPT/molopt/examples/chemxpl/MC_minimization/t.xyz"])
output = xyz_list2mols_extgraph(["./t.xyz"])

init_ncharges=[17, 17]
init_bond_orders={(0, 1) : 1}
init_cg = output[0].chemgraph 
#ChemGraph(nuclear_charges=init_ncharges, bond_orders=init_bond_orders, hydrogen_autofill=True)
#pdb.set_trace()
 
#
negcs=len(betas)

init_egcs=[ExtGraphCompound(chemgraph=deepcopy(init_cg)) for i in range(negcs)]

model_path = "/store/common/jan/"
min_func = QM9_properties(model_path=model_path, verbose=True)
min_func_name = "QM9_model"

rw=RandomWalk(bias_coeff=bias_coeff, randomized_change_params=randomized_change_params,
                            bound_enforcing_coeff=bound_enforcing_coeff, betas=betas, min_function=min_func,
                            init_egcs=init_egcs, conserve_stochiometry=True, min_function_name="QM9_model",
                            keep_histogram=True, num_saved_candidates=3)
for MC_step in range(num_MC_steps):
    rw.global_random_change(**global_change_params)
    # We can access "num_saved_candidates" of best candidates encountered so far as follows:
    print("Current best at step:", MC_step)
    for i, cc in enumerate(rw.saved_candidates):
        print(i, cc)
    print("Number of moves since last change:", rw.moves_since_changed)

print("Number of explored molecules:", len(rw.histogram))

dump2pkl(rw.histogram, "QM9_histogram.pkl")
dump2pkl(rw.saved_candidates, "QM9_best_candidates.pkl")
