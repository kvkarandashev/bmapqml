#!/bin/bash

rm -f *.pkl

rf_wdumps=restart_file_wdumps.pkl
rf_nodumps=restart_file_nodumps.pkl
rf_wdumps_fin=restart_file_wdumps_final.pkl

hdumps_prefix=histogram_dump_

python MC_dump_histogram_usage.py $rf_wdumps $hdumps_prefix > hist_size_wdumps.log
python MC_dump_histogram_usage.py $rf_nodumps > hist_size_no_dumps.log

# Merge histogram dumps with the corresponding restart file.
python ../../../misc_commands/chemxpl_related/merge_dumps_wrestart.py $rf_wdumps_fin $rf_wdumps ${hdumps_prefix}*

# Compare the two restart files.
check_traj_equivalence $rf_wdumps_fin $rf_nodumps
