from bmapqml.chemxpl.valence_treatment import str2ChemGraph
from bmapqml.chemxpl.rdkit_draw_utils import (
    draw_all_possible_resonance_structures,
    LIGHTBLUE,
    LIGHTRED,
    LIGHTGREEN,
)
from bmapqml.chemxpl.modify import FragmentPair, randomized_split_membership_vector
import numpy as np
import random

random.seed(1)
np.random.seed(1)

chemgraph_strings = [
    "6#3@1:5#1@2:6#3",
    "6#2@1:7@2:6#1@3:8",
    "6#3@1:6@2:7@3:6#1@4:6@1@5:8#1",
    "6#3@1:16@2:6@3:16@4:6#3",
    "15#2@1:6@2:15#2",
]

kwargs = {"size": (300, 200), "highlightAtomRadius": 0.4}

for str_id, cg_str in enumerate(chemgraph_strings):
    cg = str2ChemGraph(cg_str)

    file_prefix_cg = "test_cg_resonance_" + str(str_id) + "_"

    draw_all_possible_resonance_structures(cg, file_prefix_cg, **kwargs)

    file_prefix_fp = "test_fp_resonance_" + str(str_id) + "_"

    rand_memb_vec = randomized_split_membership_vector(cg, int(0.5 * cg.nhatoms()))

    fp = FragmentPair(cg, membership_vector=rand_memb_vec)

    draw_all_possible_resonance_structures(
        fp,
        file_prefix_fp,
        highlight_fragment_colors=[LIGHTBLUE, LIGHTRED],
        highlight_fragment_boundary=LIGHTGREEN,
        **kwargs
    )
