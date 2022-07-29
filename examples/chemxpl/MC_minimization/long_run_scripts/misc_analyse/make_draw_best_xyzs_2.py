from bmapqml.utils import loadpkl
from bmapqml.chemxpl.utils import write_egc2xyz, egc_with_coords
import glob, os
import numpy as np

pkl_files = glob.glob("data_*/restart_file_*.pkl")
pkl_files.sort()

traj_ncut = 50000

min_func_name = "xTB_MMFF_electrolyte"

j = 1

for pkl_file in pkl_files:
    min_val_file = (
        os.path.dirname(pkl_file)
        + os.path.basename(pkl_file).replace(".", "_")
        + ".min_vals"
    )
    cur_data = loadpkl(pkl_file)
    histogram = cur_data["histogram"]
    hist_size = 0
    best_val = None
    best_tp = None
    vals_encounters = []
    for tp in histogram:
        if (tp.first_global_MC_step_encounter is not None) and (
            tp.first_global_MC_step_encounter <= traj_ncut
        ):
            hist_size += 1
            cur_val = tp.calculated_data[min_func_name]
            vals_encounters.append((cur_val, tp.first_global_MC_step_encounter))
            if (best_val is None) or (best_val > cur_val):
                best_val = cur_val
                best_tp = tp
    vals_encounters.sort(key=lambda x: x[1])

    min_val_ubound = 0.0
    num_vals_encounters = len(vals_encounters)

    f = open(min_val_file, "w")
    #    min_vals=np.repeat(min_val_ubound, traj_ncut+1)
    ve_id = 0
    cur_min_val = min_val_ubound
    for step_id in range(traj_ncut + 1):
        while ve_id != num_vals_encounters and step_id == vals_encounters[ve_id][1]:
            cur_val = vals_encounters[ve_id][0]
            if cur_min_val > cur_val:
                cur_min_val = cur_val
            ve_id += 1
        f.write(str(step_id) + " " + str(cur_min_val) + "\n")
    #        min_vals[step_id]=cur_min_val
    f.close()

    for i in range(20):
        try:
            best_candidate_egc = egc_with_coords(best_tp.egc)
        except:
            continue
        break
    print(pkl_file, hist_size, best_tp.calculated_data)
    write_egc2xyz(best_candidate_egc, "best_candidate_" + str(j) + ".xyz")

    j += 1
