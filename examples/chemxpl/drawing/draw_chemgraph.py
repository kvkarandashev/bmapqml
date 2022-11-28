from bmapqml.chemxpl.valence_treatment import str2ChemGraph
from bmapqml.chemxpl.rdkit_draw_utils import (
    draw_chemgraph_to_file,
    draw_fragment_pair_to_file,
)
from bmapqml.chemxpl.modify import FragmentPair
import numpy as np

chemgraph_str = "6#2@1:7@2:6#1@3:8"

cg = str2ChemGraph(chemgraph_str)

draw_chemgraph_to_file(
    cg,
    "test_cg_1.png",
    size=(300, 200),
    highlightAtoms=[2, 3],
    highlightAtomColor=(0.0, 1.0, 0.0),
    highlightBondTuples=[(1, 2), (2, 3)],
    highlightBondTupleColor=(0.0, 0.0, 1.0),
    highlightAtomRadius=0.4,
)

draw_chemgraph_to_file(
    cg,
    "test_cg_2.png",
    size=(300, 200),
    highlightAtoms=[2, 3],
    highlightAtomColor=(0.0, 1.0, 0.0),
    highlight_connecting_bonds=True,
    highlightBondTuples=[(0, 1)],
    highlightBondTupleColor=(0.0, 0.0, 1.0),
    highlightAtomRadius=0.4,
)

fp = FragmentPair(cg, np.array([0, 0, 1, 1]))

draw_fragment_pair_to_file(
    fp,
    "test_frag_1.png",
    size=(300, 200),
    highlight_fragment_colors=[(0.0, 1.0, 0.0), (1.0, 0.0, 0.0)],
    highlight_fragment_boundary=None,
    highlightAtomRadius=0.4,
)

draw_fragment_pair_to_file(
    fp,
    "test_frag_2.png",
    size=(300, 200),
    highlight_fragment_colors=[(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
    highlight_fragment_boundary=(0.0, 1.0, 0.0),
    highlightAtomRadius=0.4,
)
