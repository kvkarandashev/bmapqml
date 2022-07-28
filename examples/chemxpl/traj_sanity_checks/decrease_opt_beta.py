from bmapqml.utils import loadpkl
from bmapqml.chemxpl.random_walk import ordered_trajectory
import sys

restart_file_name = sys.argv[1]

min_func_name = sys.argv[2]

num_virtual_betas = int(sys.argv[3])

restart_data = loadpkl(restart_file_name)

histogram = restart_data["histogram"]

trajectory = ordered_trajectory(histogram)

prev_min = None

for step_id, tp_list in enumerate(trajectory):
    cur_min = min(
        tp.calculated_data[min_func_name] for tp in tp_list[:num_virtual_betas]
    )
    print(step_id, cur_min)
    if prev_min is not None:
        if cur_min > prev_min:
            print("Bad step:", step_id)
            quit()
    prev_min = cur_min
