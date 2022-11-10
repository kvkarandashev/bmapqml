from bmapqml.utils import loadpkl
import sys
from bmapqml.chemxpl.ext_graph_compound import ExtGraphCompound
from bmapqml.chemxpl.utils import SMILES_to_egc, str2ChemGraph, ChemGraphStr_to_SMILES
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

test_egcs = None

if nargs < 3:
    test_SMILES = possible_test_SMILES["FLUORINATION"]
else:
    if nargs == 3:
        test_SMILES = possible_test_SMILES[sys.argv[2]]
    else:
        test_SMILES = None
        if sys.argv[2] == "--SMILES":
            test_SMILES = [sys.argv[3]]
        if sys.argv[2] == "--ChemGraph":
            cg_str = sys.argv[3]
            test_egcs = [ExtGraphCompound(chemgraph=str2ChemGraph(cg_str))]
            test_SMILES = [ChemGraphStr_to_SMILES(cg_str)]

if test_egcs is None:
    if test_SMILES is None:
        print("Wrong flag")
        quit()
    test_egcs = [SMILES_to_egc(SMILES) for SMILES in test_SMILES]

min_func = loadpkl(sys.argv[1])

for egc, SMILES in zip(test_egcs, test_SMILES):
    tp = TrajectoryPoint(egc=egc)
    print(SMILES, egc, min_func(tp))
