#!/bin/bash

num_MC_steps_1=2000

num_MC_steps_2=1000

restart_mid=restart_middle.pkl

other_final_restart="other_final_restart.pkl"

# Run a single simulation that creates a restart file during the first num_MC_steps_1 steps.
python MC_all_moves_wrestart.py $((num_MC_steps_1+num_MC_steps_2)) $restart_mid make_restart_frequency=$num_MC_steps_1

# Start a simulation from the middle restart file and make additional num_MC_steps_2.
python MC_all_moves_wrestart.py $num_MC_steps_2 $other_final_restart init_restart_file=$restart_mid

# Compare the two restart files created in the end of the simulations.
check_traj_equivalence final_restart.pkl $other_final_restart
