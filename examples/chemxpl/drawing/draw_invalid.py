from bmapqml.chemxpl.valence_treatment import str2ChemGraph
from bmapqml.chemxpl.rdkit_draw_utils import (
    draw_chemgraph_to_file,
    LIGHTRED,
    LIGHTBLUE,
)

chemgraph_str = "7#1@1:15#1@2:7#1"

cg = str2ChemGraph(chemgraph_str)

draw_chemgraph_to_file(
    cg,
    "test_invalid.png",
    size=(300, 200),
    highlightAtoms=[0, 2],
    highlightAtomColor=LIGHTRED,
    highlightBondTuples=[(0, 2)],
    highlightBondTupleColor=LIGHTBLUE,
    highlightAtomRadius=0.4,
    post_added_bonds=[(0, 2, 1)],
)
