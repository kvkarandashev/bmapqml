import numpy as np
from bmapqml.utils import loadpkl
import sys

entry_list = loadpkl(sys.argv[1])

quantities = ["energy", "HOMO_LUMO_gap", "solvation_energy", "energy_no_solvent"]

chars = ["mean", "std"]

arrs = {}

for char in chars:
    arrs[char] = []

for quant in quantities:
    arrs = {}

    for char in chars:
        arrs[char] = []

    for entry_dict in entry_list:
        if char not in entry_dict:
            continue
        for char in chars:
            cur_val = entry_dict[char][quant]
            if cur_val is not None:
                arrs[char].append(cur_val)
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
            print("FULL EXTREMA DATA " + f, arrs["mean"][i], arrs["std"][i])
