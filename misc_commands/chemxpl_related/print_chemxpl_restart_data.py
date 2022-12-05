# TODO bugged
from bmapqml.utils import loadpkl
import sys
import numpy as np
import sys

pkl_file = sys.argv[1]

cur_data = loadpkl(pkl_file)

histogram = cur_data["histogram"]

traj_ncut = int(sys.argv[2])

if len(sys.argv) > 3:
    min_func_name = sys.argv[3]
else:
    min_func_name = cur_data["min_function_name"]

num_first_encounters = np.zeros((traj_ncut + 1,), dtype=int)
discovered_min = [None for _ in range(traj_ncut + 1)]
for tp in histogram:
    cur_val = tp.calculated_data[min_func_name]
    if cur_val is None:
        continue
    if tp.first_global_MC_step_encounter <= traj_ncut:
        cur_step = tp.first_global_MC_step_encounter
        num_first_encounters[cur_step] += 1
        if (discovered_min[cur_step] is None) or (discovered_min[cur_step] > cur_val):
            discovered_min[cur_step] = cur_val

cur_hist_size = np.zeros((traj_ncut + 1,), dtype=int)
cur_hist_size[0] = num_first_encounters[0]

cur_min_val = np.empty((traj_ncut + 1,))
cur_min_val[0] = discovered_min[0]
next_min_val = cur_min_val[0]

for step_id in range(traj_ncut):
    cur_hist_size[step_id + 1] = (
        cur_hist_size[step_id] + num_first_encounters[step_id + 1]
    )
    next_disc_min = discovered_min[step_id + 1]
    if next_disc_min is not None:
        next_min_val = min(next_disc_min, next_min_val)
    cur_min_val[step_id + 1] = next_min_val

for step_id, zargs in enumerate(
    zip(cur_min_val, cur_hist_size, num_first_encounters, discovered_min)
):
    print(step_id, *zargs)
