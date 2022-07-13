from bmapqml.chemxpl.random_walk import TrajectoryPoint
from bmapqml.chemxpl.ext_graph_compound import ExtGraphCompound
from bmapqml.chemxpl.minimized_functions import (
    FF_xTB_res_dict,
    FF_xTB_dipole,
    FF_xTB_HOMO_LUMO_gap,
)
import numpy as np

nuclear_charges = np.array([9, 17])

adj_mat = np.array([[0, 1], [1, 0]])

egc = ExtGraphCompound(nuclear_charges=nuclear_charges, adjacency_matrix=adj_mat)

print(egc)

tp = TrajectoryPoint(egc=egc)

func1 = FF_xTB_res_dict(ff_type="MMFF")

print("Result dictionnary:", func1(tp))

for ff_type in ["UFF", "MMFF"]:
    tp.calculated_data = {}
    print("FF:", ff_type)
    func2 = FF_xTB_dipole(ff_type=ff_type)

    func3 = FF_xTB_HOMO_LUMO_gap(ff_type=ff_type)

    for func in [func2, func3]:
        print(func(tp))

print("Saved data:", list(dict(tp.calculated_data)))
