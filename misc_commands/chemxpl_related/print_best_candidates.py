from bmapqml.chemxpl.utils import FFInconsistent
from bmapqml.utils import write_xyz_file
from bmapqml.chemxpl.minimized_functions.morfeus_quantity_estimates import (
    morfeus_coord_info_from_tp,
)
from bmapqml.chemxpl.random_walk import CandidateCompound
from bmapqml.utils import loadpkl
from sortedcontainers import SortedList
import sys

pkl_file = sys.argv[1]

traj_ncut = int(sys.argv[2])

num_best_candidates = int(sys.argv[3])

cur_data = loadpkl(pkl_file)
if len(sys.argv) > 4:
    min_func_name = sys.argv[4]
else:
    min_func_name = cur_data["min_function_name"]

best_candidates = SortedList()


for entry in cur_data["histogram"]:
    if entry.first_global_MC_step_encounter <= traj_ncut:
        cur_val = entry.calculated_data[min_func_name]
        if cur_val is not None:
            cur_cc = CandidateCompound(entry, entry.calculated_data[min_func_name])
            best_candidates.add(cur_cc)
            while len(best_candidates) > num_best_candidates:
                del best_candidates[-1]

for cand_id, cand in enumerate(best_candidates):
    xyz_name = "best_candidate_" + str(cand_id) + ".xyz"
    extra_string = ""
    tp = cand.tp
    for val_name, val in tp.calculated_data.items():
        if isinstance(val, float):
            extra_string += val_name + "=" + str(val) + " "
    try:
        coord_info = morfeus_coord_info_from_tp(tp)
        write_xyz_file(
            coord_info["coordinates"],
            xyz_name,
            nuclear_charges=coord_info["nuclear_charges"],
            extra_string=extra_string[:-1],
        )
    except FFInconsistent:
        print(cand_id, tp, cand.func_val)
