from bmapqml.test_utils import dirs_xyz_list
from bmapqml.chemxpl.utils import xyz2mol_extgraph
from bmapqml.chemxpl.random_walk import TrajectoryPoint
from bmapqml.chemxpl.minimized_functions.xtb_quantity_estimates import (
    FF_xTB_HOMO_LUMO_gap,
    FF_xTB_dipole,
)
from bmapqml.chemxpl.minimized_functions import LinearCombination
import os, copy
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


def quant_est_wtp(tp, quant_estimate):
    v = quant_estimate(tp)
    return [v, tp]


def est_stddev_vals(quant_estimate, tp_array):
    quant_vals_tps = embarrassingly_parallel(quant_est_wtp, tp_array, (quant_estimate,))
    quant_vals = []
    final_quant_vals = []
    for i, val_tp in enumerate(quant_vals_tps):
        v = val_tp[0]
        if v is not None:
            final_quant_vals.append(v)
        quant_vals.append(v)
        tp_array[i].calculated_data = copy.deepcopy(val_tp[1].calculated_data)
    final_quant_vals = np.array(final_quant_vals)
    return np.std(final_quant_vals), quant_vals


quant_signs = [-1, -1]

quant_estimate_types = [FF_xTB_dipole, FF_xTB_HOMO_LUMO_gap]
quant_names = ["Dipole", "HOMO_LUMO_gap"]

ff_types = ["MMFF", "UFF"]

for ff_type in ff_types:
    est_kwargs = {
        "num_ff_attempts": num_ff_attempts,
        "ff_type": ff_type,
        "display_problematic_xTB_tps": True,
    }
    quant_estimates = [
        quant_estimate_type(**est_kwargs)
        for quant_estimate_type in quant_estimate_types
    ]
    for tp_id in range(len(tps)):
        tps[tp_id].calculated_data = {}
    est_vals = {"xyzs": xyz_list}
    coeffs = []
    for quant_sign, quant_estimate, quant_name in zip(
        quant_signs, quant_estimates, quant_names
    ):
        est_stddev, est_vals[quant_name] = est_stddev_vals(quant_estimate, tps)
        coeffs.append(quant_sign / est_stddev)
    # Added for testing purposes
    for tp, xyz, v1, v2 in zip(
        tps, est_vals["xyzs"], est_vals[quant_names[0]], est_vals[quant_names[1]]
    ):
        if (None in [v1, v2]) and (v1 is not v2):
            print("Failed at:", xyz, v1, v2)
            print("Calculated data:", tp.calculated_data)
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
