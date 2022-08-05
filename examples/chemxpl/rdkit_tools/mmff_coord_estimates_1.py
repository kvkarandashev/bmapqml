from bmapqml.chemxpl.valence_treatment import ChemGraph
from bmapqml.chemxpl.ext_graph_compound import ExtGraphCompound
from bmapqml.chemxpl.utils import egc_with_coords, write_egc2xyz

nuclear_charges1 = [6, 7, 6]
bond_orders1 = {(0, 1): 1, (1, 2): 1}

nuclear_charges2 = [7, 6, 6]
bond_orders2 = {(0, 1): 1, (0, 2): 1}

for mol_id, (nuclear_charges, bond_orders) in enumerate(
    zip([nuclear_charges1, nuclear_charges2], [bond_orders1, bond_orders2])
):

    cg = ChemGraph(
        nuclear_charges=nuclear_charges, bond_orders=bond_orders, hydrogen_autofill=True
    )
    egc1 = ExtGraphCompound(chemgraph=cg)
    print(egc1)
    egc2 = egc_with_coords(egc1, pick_minimal_conf=True)
    print(egc2.coordinates)
    print(egc2.additional_data["canon_rdkit_SMILES"])
    print(egc2.additional_data["canon_rdkit_heavy_atom_index"])
    print(egc2.additional_data["canon_rdkit_hydrogen_connection"])
    write_egc2xyz(egc2, "mmff_gen_coords_" + str(mol_id) + ".xyz")
