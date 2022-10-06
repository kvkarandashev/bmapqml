from bmapqml.utils import loadpkl
import sys
from bmapqml.chemxpl.utils import SMILES_to_egc
from bmapqml.chemxpl.random_walk import TrajectoryPoint

nargs = len(sys.argv)

if nargs < 2:
    print("Minimized function pkl file should be the second argument.")
    quit()

possible_test_SMILES = {
    "FLUORINATION": [
        "C",
        "CF",
        "FCF",
        "FC(F)F",
        "FC(F)(F)F",
    ],
    "ENOLS": [
        "C1=CC=CC=C1",
        "C1=CC=C(O)C=C1",
        "C=O",
        "C=C",
        "OC=C",
        "OCO",
        "N1C(O)=C(O)C(O)=C(O)1",
        "N1C=CC=C1",
    ],
}

if nargs < 3:
    test_SMILES = possible_test_SMILES["FLUORINATION"]
else:
    if nargs == 3:
        test_SMILES = possible_test_SMILES[sys.argv[2]]
    else:
        if sys.argv[2] == "--SMILES":
            test_SMILES = [sys.argv[3]]

min_func = loadpkl(sys.argv[1])

for SMILES in test_SMILES:
    tp = TrajectoryPoint(egc=SMILES_to_egc(SMILES))
    print(SMILES, min_func(tp))
