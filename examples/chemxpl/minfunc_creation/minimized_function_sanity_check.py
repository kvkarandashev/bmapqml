# The script takes a string representation of a molecular graph and evaluates the minimized function for it.
# If repeating this script several times yields inconsistent result for the same SMILES then the function should be fixed.
import sys
from bmapqml.utils import loadpkl
from bmapqml.chemxpl.utils import SMILES_to_egc
from bmapqml.chemxpl.random_walk import TrajectoryPoint

SMILES_str = None
minfunc_location = None

if len(sys.argv) > 3:
    minfunc_location = sys.argv[3]
    if len(sys.argv[1] == "--SMILES"):
        SMILES_str = sys.argv[2]
else:
    SMILES_str = sys.argv[1]
    minfunc_location = sys.argv[2]

est_func = loadpkl(minfunc_location)

tp = TrajectoryPoint(egc=SMILES_to_egc(SMILES_str))

print(est_func(tp))
