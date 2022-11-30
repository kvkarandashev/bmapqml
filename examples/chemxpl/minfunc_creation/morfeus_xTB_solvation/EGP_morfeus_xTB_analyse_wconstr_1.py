import numpy as np
from bmapqml.utils import loadpkl, dump2pkl
from bmapqml.chemxpl.ext_graph_compound import ExtGraphCompound
from bmapqml.chemxpl.valence_treatment import str2ChemGraph, sorted_tuple
from bmapqml.chemxpl.random_walk import egc_valid_wrt_change_params
from bmapqml.chemxpl.test_utils import elements_in_egc_list, egc_list_nhatoms_hist
from bmapqml.chemxpl.minimized_functions.morfeus_quantity_estimates import (
    LinComb_Morfeus_xTB_code,
)
import sys
from sortedcontainers import SortedDict, SortedList

gap_name = "HOMO_LUMO_gap"


def print_data(egcs, add_line):
    print("DATA FOR:", add_line)
    num_egcs = len(egcs)
    print("Number of entries:", num_egcs)
    print("Size histogram:")
    num_up_to = 0
    for s, n in egc_list_nhatoms_hist(egcs).items():
        num_up_to += n
        print(s, ":", n, num_up_to / num_egcs)
    print("Available elements:", elements_in_egc_list(egcs))


def compliant_egc_list(egcs, gap_constraint=None, **other_constraints):
    output = []
    for egc in egcs:
        if egc_valid_wrt_change_params(egc, **other_constraints):
            if gap_constraint is not None:
                if egc.additional_data["mean"][gap_name] < gap_constraint:
                    continue
            output.append(egc)
    return output


def print_present_bonds(egc_list):
    max_bos = SortedDict()
    max_bo_examples = SortedDict()
    for egc in egc_list:
        cg = egc.chemgraph
        for bt in cg.bond_orders.keys():
            ncharge_tuple = sorted_tuple(
                cg.hatoms[bt[0]].ncharge, cg.hatoms[bt[1]].ncharge
            )
            max_order = max(cg.aa_all_bond_orders(*bt))
            if ncharge_tuple in max_bos:
                max_bos[ncharge_tuple] = max(max_bos[ncharge_tuple], max_order)
            else:
                max_bos[ncharge_tuple] = max_order
                max_bo_examples[ncharge_tuple] = SortedDict()
            max_bo_examples[ncharge_tuple][max_order] = cg
    print("Found bond orders:")
    for btuple, morder in max_bos.items():
        print(btuple, morder)
    print("Found examples:")
    for btuple, example_dict in max_bo_examples.items():
        print("ncharge tuple", btuple, "examples:")
        for bo, example in example_dict.items():
            print(bo, example)
    print("Finished")


def print_with_element(egc_list, checked_ncharge):
    for egc in egc_list:
        if checked_ncharge in egc.nuclear_charges:
            print(egc)


entry_list = loadpkl(sys.argv[1])

if len(sys.argv) < 3:
    solvent = "water"
else:
    solvent = sys.argv[2]

print("Total number of entries:", len(entry_list))

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

# Which quantities are used to create minimized functions.
minfunc_base_quantities = ["dipole", "solvation_energy"]

# Which sign a quantity is included with.
minfunc_quant_signs = {"dipole": -1, "solvation_energy": 1}


# First of all, print size and composition analysis of compounds for which the calculations converged.

# print(entry_list[0])
# quit()

valid_entries = []

for entry in entry_list:
    if entry["arrs"]["energy"] is None:
        continue
    if entry["mean"]["energy"] is None:
        continue
    valid_entries.append(entry)

egcs = SortedList()

for entry in valid_entries:
    cg_str = entry["chemgraph_str"]
    cg = str2ChemGraph(cg_str)
    egcs.add(ExtGraphCompound(chemgraph=cg, additional_data=entry))

print_data(egcs, "ENTRIES WITH CONVERGED CALCULATIONS")

size_constraints = {"nhatoms_range": [1, 15]}

bond_constraints = {
    "not_protonated": [5, 8, 9, 14, 15, 16, 17, 35],
    "forbidden_bonds": [
        (7, 7),
        (8, 8),
        (9, 9),
        (7, 8),
        (7, 9),
        (8, 9),
        (7, 17),
        (8, 17),
        (9, 17),
        (17, 17),
        (7, 35),
        (8, 35),
        (9, 35),
        (17, 35),
        (35, 35),
    ],
}  # , "possible_elements" : ["B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br"]}

gap_constraints = {"weak": 0.09126989358754387, "strong": 0.17735152497325582}

present_bonds = {}

for constr_name, constr_val in gap_constraints.items():
    print("GAP CONSTRAINT ", constr_name, ":", constr_val)

    bond_constr_compl_egcs = compliant_egc_list(
        egcs, gap_constraint=constr_val, **bond_constraints
    )

    print_data(
        bond_constr_compl_egcs, "BOND AND HOMO-LUMO CONSTRAINT COMPLIANT ENTRIES"
    )

    print("PRESENT BONDS:")
    print_present_bonds(bond_constr_compl_egcs)

    print("PRESENT CHEMGRAPHS WITH BORON:")
    print_with_element(bond_constr_compl_egcs, 5)

    constr_compl_egcs = compliant_egc_list(bond_constr_compl_egcs, **size_constraints)

    print_data(constr_compl_egcs, "SIZE AND BOND CONSTRAINT COMPLIANT ENTRIES")

    chars = ["mean", "std"]

    chemgraph_strs = []

    arrs = {}

    for char in chars:
        arrs[char] = []

    for quant in quantities:
        arrs = {}

        for char in chars:
            arrs[char] = []

        for egc in constr_compl_egcs:
            for char in chars:
                cur_val = egc.additional_data[char][quant]
                arrs[char].append(cur_val)

        print("Quantity", quant)

        extrema = {}

        for char in chars:
            print(char, ":")
            arr = arrs[char]
            extrema[char] = {"MIN": np.argmin(arr), "MAX": np.argmax(arr)}

            stddev = np.std(arr)

            print("MIN", min(arr))
            print("MAX", max(arr))
            print("MEAN", np.mean(arr))
            print("STD", stddev)

            for f, i in extrema[char].items():
                print(
                    "FULL EXTREMA DATA " + f,
                    arrs["mean"][i],
                    arrs["std"][i],
                    constr_compl_egcs[i],
                )  # , constr_compl_egcs[i].additional_data)

            if (char == "mean") and (quant in minfunc_base_quantities):
                min_func = LinComb_Morfeus_xTB_code(
                    [quant],
                    [minfunc_quant_signs[quant] / stddev],
                    constr_quants=[gap_name],
                    cq_lower_bounds=[constr_val],
                    num_attempts=1,
                    num_conformers=8,
                    ff_type="MMFF94",
                    solvent=solvent,
                )
                pkl_name = (
                    "egp_morfeus_xTB_"
                    + solvent
                    + "_"
                    + quant
                    + "_"
                    + constr_name
                    + ".pkl"
                )
                dump2pkl(min_func, pkl_name)
