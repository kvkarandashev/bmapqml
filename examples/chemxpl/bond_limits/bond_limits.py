# The script tests how the code putting the 
from bmapqml.chemxpl.random_walk import TrajectoryPoint
from bmapqml.chemxpl.valence_treatment import ChemGraph

nuclear_charges=[15, 15, 6, 6]
bond_order_dicts=[{(0, 1) : 3, (1, 2) : 2, (0, 3) : 2, (2, 3) : 2},
                   {(0, 1) : 3, (1, 2) : 1, (0, 3) : 1, (2, 3) : 1},
                    {(0, 1) : 3, (1, 2) : 1, (0, 3) : 1, (2, 3) : 2}]
cgs=[ChemGraph(nuclear_charges=nuclear_charges, bond_orders=bond_order_dict, hydrogen_autofill=True) for bond_order_dict in bond_order_dicts]

tp=TrajectoryPoint(cg=cgs[1])

for cg in cgs:
    print("Mol:", cg)
    print("Resonance structures:", cg.resonance_structure_orders)
    tp=TrajectoryPoint(cg=cg)
    tp.init_possibility_info()
    print(tp.possibilities())
