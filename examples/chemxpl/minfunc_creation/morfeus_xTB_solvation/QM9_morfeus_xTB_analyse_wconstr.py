import numpy as np
from bmapqml.utils import loadpkl
from bmapqml.chemxpl.valence_treatment import str2ChemGraph
from bmapqml.chemxpl.minimized_functions.mol_constraints import (
    NoProtonation,
    NoForbiddenBonds,
)
from bmapqml.chemxpl.random_walk import TrajectoryPoint
import sys

entry_list = loadpkl(sys.argv[1])

quantities = [
    "dipole",
    "energy",
    "HOMO_LUMO_gap",
    "solvation_energy",
    "energy_no_solvent",
    "atomization_energy",
    "normalized_atomization_energy",
    "num_evals",
]

chars = ["mean", "std"]

no_prot = [8, 9]

forbidden_bonds = [(7, 7), (8, 8), (9, 9), (7, 8), (7, 9), (8, 9)]


xyzs = []

constraints = [
    NoProtonation(restricted_ncharges=no_prot),
    NoForbiddenBonds(forbidden_bonds=forbidden_bonds),
]


def constraints_broken(tp):
    for constr in constraints:
        if not constr(tp):
            return True
    return False


arrs = {}

for char in chars:
    arrs[char] = []

for quant in quantities:
    arrs = {}

    for char in chars:
        arrs[char] = []

    for entry_dict in entry_list:
        cur_xyz = entry_dict["xyz"]

        if char not in entry_dict:
            continue

        tp = TrajectoryPoint(cg=str2ChemGraph(entry_dict["chemgraph"]))

        if constraints_broken(tp):
            continue

        for char in chars:
            cur_val = entry_dict[char][quant]
            if cur_val is not None:
                arrs[char].append(cur_val)
                if (char == chars[0]) and (quant == quantities[0]):
                    xyzs.append(cur_xyz)
    print()
    print("Quantity", quant)

    extrema = {}

    for char in chars:
        print(char, ":")
        arr = arrs[char]
        extrema[char] = {"MIN": np.argmin(arr), "MAX": np.argmax(arr)}

        print("MIN", min(arr))
        print("MAX", max(arr))
        print("MEAN", np.mean(arr))
        print("STD", np.std(arr))

        for f, i in extrema[char].items():
            print("FULL EXTREMA DATA " + f, arrs["mean"][i], arrs["std"][i], xyzs[i])
