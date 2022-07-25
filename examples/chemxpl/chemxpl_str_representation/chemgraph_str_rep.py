from bmapqml.chemxpl.valence_treatment import ChemGraph, str2ChemGraph
import numpy as np

ncharges1 = [6, 7]

ncharges2 = [7, 6]

ncharges3 = [7, 6, 6]

ncharges4 = [6, 7, 6]

ncharges5 = [6]

ncharges6 = [9, 9]

all_ncharges = [ncharges1, ncharges2, ncharges3, ncharges4, ncharges5, ncharges6]

bond_orders1 = {(0, 1): 1}

bond_orders2 = {(0, 1): 1}

bond_orders3 = {(0, 1): 1, (1, 2): 1}

bond_orders4 = {(0, 1): 1, (0, 2): 1}

bond_orders5 = {}

bond_orders6 = {(0, 1): 1}

all_bond_orders = [
    bond_orders1,
    bond_orders2,
    bond_orders3,
    bond_orders4,
    bond_orders5,
    bond_orders6,
]

for ncharges, bond_orders in zip(all_ncharges, all_bond_orders):
    cg = ChemGraph(
        nuclear_charges=ncharges, bond_orders=bond_orders, hydrogen_autofill=True
    )

    print(cg)

    cg_str = str(cg)

    cg_conv = str2ChemGraph(cg_str)
    print(cg_conv == cg)
