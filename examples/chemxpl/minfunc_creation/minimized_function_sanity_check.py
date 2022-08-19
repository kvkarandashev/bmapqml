# The script takes a string representation of a molecular graph and evaluates the minimized function for it.
# If repeating this script several times yields inconsistent result for the same SMILES then the function should be fixed.
# Example of usage:
# python minimized_function_sanity_check.py C /store/common/konst/chemxpl_related/minimized_function_xTB_MMFF_min_en_conf_electrolyte.pkl
import sys
from bmapqml.utils import loadpkl
from bmapqml.chemxpl.utils import SMILES_to_egc
from bmapqml.chemxpl.random_walk import TrajectoryPoint
from bmapqml.chemxpl.valence_treatment import str2ChemGraph

SMILES_str = None
chemgraph_str = None
minfunc_location = sys.argv[-1]

if len(sys.argv) > 3:
    if len(sys.argv[1] == "--SMILES"):
        SMILES_str = sys.argv[2]
    elif len(sys.argv[1] == "--chemgraph_str"):
        chemgraph_str = sys.argv[2]
else:
    SMILES_str = sys.argv[1]

egc = None

if SMILES_str is not None:
    egc = SMILES_to_egc(SMILES_str)

if chemgraph_str is not None:
    egc = str2ChemGraph(chemgraph_str)

if egc is None:
    print("ERROR")
    quit()

if len(sys.argv) < 4:
    num_attempts = 1
else:
    num_attempts = int(sys.argv[3])

est_func = loadpkl(minfunc_location)

for _ in range(num_attempts):
    tp = TrajectoryPoint(egc=egc)

    print(est_func(tp))
