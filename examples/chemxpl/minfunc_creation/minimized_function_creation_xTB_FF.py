from bmapqml.test_utils import dirs_xyz_list
from bmapqml.chemxpl.utils import xyz2mol_extgraph
from bmapqml.chemxpl.random_walk import TrajectoryPoint
from bmapqml.chemxpl.minimized_functions import (
    FF_xTB_HOMO_LUMO_gap,
    FF_xTB_dipole,
    LinearCombination,
)
import os
from bmapqml.utils import dump2pkl, embarrassingly_parallel
import numpy as np


def xyz2tp(xyz_name):
    egc = xyz2mol_extgraph(xyz_name)
    return TrajectoryPoint(egc=egc)


def all_xyzs2tps(xyz_list):
    return embarrassingly_parallel(xyz2tp, xyz_list, ())


num_ff_attempts = 10

QM9_xyz_dir = os.environ["DATA"] + "/QM9_filtered/xyzs"

pkl_store_dir = "/store/common/konst/chemxpl_related"

xyz_list = dirs_xyz_list(QM9_xyz_dir)

tps = all_xyzs2tps(xyz_list)


def est_stddev_vals(quant_estimate, tp_array):
    quant_vals = embarrassingly_parallel(quant_estimate, tp_array, ())
    final_quant_vals = []
    for qv in quant_vals:
        if qv is not None:
            final_quant_vals.append(qv)
    final_quant_vals = np.array(final_quant_vals)
    return np.std(final_quant_vals), quant_vals


quant_signs = [-1, -1]

quant_estimate_types = [FF_xTB_dipole, FF_xTB_HOMO_LUMO_gap]
quant_names = ["Dipole", "HOMO_LUMO_gap"]

ff_types = ["MMFF", "UFF"]

for ff_type in ff_types:
    est_kwargs = {"num_ff_attempts": num_ff_attempts, "ff_type": ff_type}
    quant_estimates = [
        quant_estimate_type(**est_kwargs)
        for quant_estimate_type in quant_estimate_types
    ]
    for tp_id in range(len(xyz_list)):
        tps[tp_id].calculated_data = {}
    est_vals = {"xyzs": xyz_list}
    coeffs = []
    for quant_sign, quant_estimate, quant_name in zip(
        quant_signs, quant_estimates, quant_names
    ):
        est_stddev, est_vals[quant_name] = est_stddev_vals(quant_estimate, tps)
        coeffs.append(quant_sign / est_stddev)
    min_function = LinearCombination(quant_estimates, coeffs, quant_names)

    print("Testing function output:", min_function(tps[-1]))

    dump2pkl(
        min_function,
        pkl_store_dir + "/minimized_function_xTB_" + ff_type + "_electrolyte.pkl",
    )
    dump2pkl(
        est_vals,
        pkl_store_dir + "/QM9_data_min_func_xTB_" + ff_type + "_electrolyte.pkl",
    )
