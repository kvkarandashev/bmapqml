#!/bin/bash

num_MC_steps=2000

num_components=2

# Run $num_components simulations with one restarting from the other.
for component_id in $(seq $num_components)
do
    new_restart=restart_$component_id.pkl
    python MC_all_moves_wrestart.py $num_MC_steps $new_restart $prev_restart
    prev_restart=$new_restart
done
# Run one long simulation with no restarts.
python MC_all_moves_wrestart.py $((num_MC_steps*num_components)) restart_no_pause.pkl

# Compare the two restart files.
python check_traj_equivalence.py restart_no_pause.pkl $new_restart
