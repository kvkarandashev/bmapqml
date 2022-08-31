from bmapqml.chemxpl.random_walk import TrajectoryPoint
from bmapqml.chemxpl.ext_graph_compound import ExtGraphCompound
from bmapqml.chemxpl.minimized_functions.morfeus_quantity_estimates import (
    morfeus_FF_xTB_dipole,
    morfeus_FF_xTB_HOMO_LUMO_gap,
    morpheus_coord_info_from_tp,
)
import numpy as np

num_ff_attempts = 5

nuclear_charges = np.array([7])

adj_mat = np.array([[0]])

egc = ExtGraphCompound(
    nuclear_charges=nuclear_charges, adjacency_matrix=adj_mat, hydrogen_autofill=True
)

print(egc)

tp = TrajectoryPoint(egc=egc)

print(
    "Coordinate info:",
    morpheus_coord_info_from_tp(tp, num_attempts=num_ff_attempts, ff_type="MMFF94"),
)

for ff_type in ["MMFF94", "MMFF94s"]:
    print("FF type:", ff_type)
    func_init_kwargs = {
        "ff_type": ff_type,
        "num_ff_attempts": num_ff_attempts,
    }
    tp.calculated_data = {}
    func1 = morfeus_FF_xTB_dipole(**func_init_kwargs)

    func2 = morfeus_FF_xTB_HOMO_LUMO_gap(**func_init_kwargs)

    names = ["Dipole", "HOMO_LUMO_gap"]

    for func, name in zip([func1, func2], names):
        print(name, ":", func(tp))
    coords = tp.calculated_data["coord_info"]["coordinates"]
    print("Bond length:", np.sqrt(np.sum((coords[0] - coords[1]) ** 2)))

print("Saved data:", list(dict(tp.calculated_data)))
