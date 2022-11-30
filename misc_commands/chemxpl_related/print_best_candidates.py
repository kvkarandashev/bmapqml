from bmapqml.chemxpl.utils import FFInconsistent
from bmapqml.utils import write_xyz_file
from bmapqml.chemxpl.minimized_functions.morfeus_quantity_estimates import (
    morfeus_coord_info_from_tp,
)
from bmapqml.chemxpl.random_walk import CandidateCompound
from bmapqml.utils import loadpkl
from bmapqml.chemxpl.rdkit_draw_utils import draw_chemgraph_to_file
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

file_prefix = "best_candidate_"

for cand_id, cand in enumerate(best_candidates):
    file_basename = "best_candidate_" + str(cand_id)
    xyz_name = file_basename + ".xyz"
    image_name = file_basename + ".png"

    extra_string = ""
    tp = cand.tp
    draw_chemgraph_to_file(
        tp.egc.chemgraph, image_name, size=(600, 400), bw_palette=False
    )
    for val_name, val in tp.calculated_data.items():
        if isinstance(val, float):
            extra_string += val_name + "=" + str(val) + " "

    coord_info = morfeus_coord_info_from_tp(tp, num_attempts=128)
    coords = coord_info["coordinates"]
    if coords is None:
        xyz_output = open(xyz_name, "w")
        print(cand_id, tp, cand.func_val, file=xyz_output)
        xyz_output.close()
    else:
        write_xyz_file(
            coords,
            xyz_name,
            nuclear_charges=coord_info["nuclear_charges"],
            extra_string=extra_string[:-1],
        )
