#!/bin/bash

rm -f *.pkl

num_MC_steps_1=2000

num_MC_steps_2=3000

final_restart="final_restart.pkl"

other_final_restart="other_final_restart.pkl"

# Run a single simulation that creates a restart file during the first num_MC_steps_1 steps.
python MC_all_moves_wrestart.py $final_restart

# Start a simulation from the middle restart file and complete the simulation num_MC_steps_2.
python MC_all_moves_wrestart.py $other_final_restart

# Compare the two restart files created in the end of the simulations.
check_traj_equivalence $final_restart $other_final_restart
