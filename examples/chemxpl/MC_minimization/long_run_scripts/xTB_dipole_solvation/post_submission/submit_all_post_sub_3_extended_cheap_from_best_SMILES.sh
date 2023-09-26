#!/bin/bash

# The minimized function file is created by scripts in examples/chemxpl/minfunc_creation/morfeus_xTB_solvation.
dump_directory="/store/konst/chemxpl_related/minimization_runs_xTB_dipole_solvation_cheap_post_sub_best_SMILES"

gap_constraint="strong"

quantity="solvation_energy"

bias_strength="none"

job_name=$1

if [ "$job_name" == "" ]
then
    echo "Job name should be second argument"
    exit
fi

cur_directory=$(pwd)

py_script=$cur_directory/MC_opt_3_best_SMILES.py

full_job_name=${job_name}_${bias_strength}_${gap_constraint}_${quantity}
min_func_file=$cur_directory/../qm9_morfeus_xTB_cheap_water_${quantity}_${gap_constraint}.pkl
final_dump_directory=$dump_directory/${full_job_name}

mkdir -p $final_dump_directory
cd $final_dump_directory
for seed in $(seq 1 8)
do
    spython --OMP_NUM_THREADS=1 --CPUs=2 $py_script ${full_job_name}_$seed $min_func_file min_$quantity $seed $bias_strength
done