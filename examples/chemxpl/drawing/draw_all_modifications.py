from bmapqml.chemxpl.valence_treatment import str2ChemGraph
from bmapqml.chemxpl.rdkit_draw_utils import (
    draw_all_modification_possibilities,
    LIGHTBLUE,
    LIGHTRED,
    LIGHTGREEN,
)
from bmapqml.utils import mkdir
import numpy as np
import os

chemgraph_strings = [
    "6#3@1:8@2:6#3",
    "6#2@1:7@2:6#1@3:8",
    "6#3@1:6@2:7@3:6#1@4:6@1@5:8#1",
    "6#3@1:16@2:6@3:16@4:6#3",
    "15#2@1:5#1@2:15#2",
    "7#1@1:15#1@2:7#1",
]

kwargs = {
    "size": (300, 200),
    "highlightAtomRadius": 0.4,
    "color_change": LIGHTRED,
    "color_change_neighbors": LIGHTBLUE,
}

for str_id, cg_str in enumerate(chemgraph_strings):
    dump_dir = "mod_possibilities_" + str(str_id)
    mkdir("mod_possibilities_" + str(str_id))
    os.chdir(dump_dir)

    cg = str2ChemGraph(cg_str)

    file_prefix = "test_mod_cg_" + str(str_id) + "_"

    draw_all_modification_possibilities(cg, file_prefix, **kwargs)

    os.chdir("..")
