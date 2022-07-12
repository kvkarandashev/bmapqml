from bmapqml.chemxpl.random_walk import TrajectoryPoint
from bmapqml.chemxpl.ext_graph_compound import ExtGraphCompound
from bmapqml.chemxpl.minimized_functions import (
    MMFF_xTB_res_dict,
    MMFF_xTB_dipole,
    MMFF_xTB_HOMO_LUMO_gap,
)
import numpy as np

nuclear_charges = np.array([9, 17])

adj_mat = np.array([[0, 1], [1, 0]])

egc = ExtGraphCompound(nuclear_charges=nuclear_charges, adjacency_matrix=adj_mat)

print(egc)

tp = TrajectoryPoint(egc=egc)

func1 = MMFF_xTB_res_dict()

func2 = MMFF_xTB_dipole()

func3 = MMFF_xTB_HOMO_LUMO_gap()

for func in [func1, func2, func3]:
    print(func(tp))

print(list(dict(tp.calculated_data)))
