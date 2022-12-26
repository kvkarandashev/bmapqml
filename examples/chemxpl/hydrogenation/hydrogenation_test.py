from bmapqml.chemxpl.valence_treatment import str2ChemGraph
from bmapqml.chemxpl.random_walk import TrajectoryPoint
from bmapqml.chemxpl.minimized_functions.morfeus_quantity_estimates import (
    Hydrogenation,
    Hydrogenation_xTB,
)

funcs = [
    Hydrogenation(num_attempts=2, num_conformers=8),
    Hydrogenation_xTB(
        xTB_related_kwargs={"num_attempts": 2, "num_conformers": 8},
        xTB_related_overconverged_kwargs={"num_attempts": 8, "num_conformers": 8},
    ),
]

cg_str_list = [
    "6#4",
    "9#1",
    "6#3@1:6#3",
    "6#3@1:7#2",
    "6#3@1:8#1",
    "6#3@1:9",
    "9@1:6#2@2:6#2@3:9",
]

for cg_str in cg_str_list:
    cur_cg = str2ChemGraph(cg_str)
    cur_tp = TrajectoryPoint(cg=cur_cg)
    print(cg_str, *[func(cur_tp) for func in funcs])
