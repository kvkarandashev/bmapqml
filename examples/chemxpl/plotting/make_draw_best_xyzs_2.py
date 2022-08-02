from bmapqml.utils import loadpkl
from bmapqml.chemxpl.utils import write_egc2xyz, egc_with_coords
import glob, os, sys
import numpy as np

if len(sys.argv) == 1:
    pkl_folders = "data_*"
    output_folder = "."
else:
    pkl_folders = sys.argv[1]
    output_folder = pkl_folders


pkl_files = glob.glob(pkl_folders + "/restart_file_*.pkl")
pkl_files.sort()

traj_ncut = 100000

min_func_name = "xTB_MMFF_electrolyte"

j = 1

for pkl_file in pkl_files:
    min_val_file = (
        os.path.dirname(pkl_file)
        + "/"
        + os.path.basename(pkl_file).replace(".", "_")
        + ".expl_data"
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
    cur_hist_size = 0
    for step_id in range(traj_ncut + 1):
        while ve_id != num_vals_encounters and step_id == vals_encounters[ve_id][1]:
            cur_val = vals_encounters[ve_id][0]
            cur_hist_size += 1
            if cur_min_val > cur_val:
                cur_min_val = cur_val
            ve_id += 1
        f.write(str(step_id) + " " + str(cur_min_val) + " " + str(cur_hist_size) + "\n")
    #        min_vals[step_id]=cur_min_val
    f.close()

    for i in range(20):
        try:
            best_candidate_egc = egc_with_coords(best_tp.egc)
        except:
            continue
        break
    print(pkl_file, hist_size, best_tp.calculated_data)
    write_egc2xyz(
        best_candidate_egc, output_folder + "/best_candidate_" + str(j) + ".xyz"
    )

    j += 1
