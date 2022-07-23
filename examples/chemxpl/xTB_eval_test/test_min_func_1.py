from bmapqml.chemxpl.random_walk import TrajectoryPoint
from bmapqml.chemxpl.ext_graph_compound import ExtGraphCompound
from bmapqml.chemxpl.minimized_functions import (
    FF_xTB_res_dict,
    FF_xTB_dipole,
    FF_xTB_HOMO_LUMO_gap,
)
import numpy as np

nuclear_charges = np.array([6])

adj_mat = np.array([[0]])

egc = ExtGraphCompound(
    nuclear_charges=nuclear_charges, adjacency_matrix=adj_mat, hydrogen_autofill=True
)

print(egc)

tp = TrajectoryPoint(egc=egc)

func1 = FF_xTB_res_dict(ff_type="MMFF")

print("Result dictionnary:", func1(tp))

ff_types = {"RDKit": ["UFF", "MMFF"], "Leruli": [None]}

for coord_calculation_type in ["RDKit", "Leruli"]:
    print("Coordinates calculated with:", coord_calculation_type)

    for ff_type in ff_types[coord_calculation_type]:
        print("FF type:", ff_type)
        func_init_kwargs = {
            "ff_type": ff_type,
            "coord_calculation_type": coord_calculation_type,
        }
        tp.calculated_data = {}
        func2 = FF_xTB_dipole(**func_init_kwargs)

        func3 = FF_xTB_HOMO_LUMO_gap(**func_init_kwargs)

        names = ["Dipole", "HOMO_LUMO_gap"]

        for func, name in zip([func2, func3], names):
            print(name, ":", func(tp))
        coords = tp.calculated_data["coord_info"]["coordinates"]
        print("Bond length:", np.sqrt(np.sum((coords[0] - coords[1]) ** 2)))

print("Saved data:", list(dict(tp.calculated_data)))
