from bmapqml.chemxpl.utils import write_egc2xyz, egc_with_coords, FFInconsistent
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
    if (entry.first_global_MC_step_encounter is not None) and (
        entry.first_global_MC_step_encounter <= traj_ncut
    ):
        cur_cc = CandidateCompound(entry, entry.calculated_data[min_func_name])
        best_candidates.add(cur_cc)
        while len(best_candidates) > num_best_candidates:
            del best_candidates[-1]

for cand_id, cand in enumerate(best_candidates):
    xyz_name = "best_candidate_" + str(cand_id) + ".xyz"
    try:
        egc_wcoords = egc_with_coords(cand.tp.egc)
        write_egc2xyz(
            egc_wcoords, xyz_name, extra_string=min_func_name + "=" + str(cand.func_val)
        )
    except FFInconsistent:
        print(cand_id, cand.tp, cand.func_val)
