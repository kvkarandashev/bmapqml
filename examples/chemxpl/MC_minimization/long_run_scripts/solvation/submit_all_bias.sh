#!/bin/bash

# The minimized function file is created by scripts in examples/chemxpl/minfunc_creation, rmg_solvation and morfeus_xTB_solvation subdirectories.
dump_directory="/store/common/konst/chemxpl_related/minimization_runs_solvation"

min_func_names=("RMGNormalizedSolvation_8" "constr_solvation_morfeus_xTB_data_water")

job_name=$1

if [ "$job_name" == "" ]
then
    echo "Job name should be second argument"
    exit
fi

cur_directory=$(pwd)

py_script=$cur_directory/MC_bias.py


for min_func_name in ${min_func_names[@]}
do
    if [ "$min_func_name" == "constr_solvation_morfeus_xTB_data_water" ]
    then
        env=base
        CPUs=2
    else
        env=rmg_env_fresh
        CPUs=1
    fi

    final_dump_directory=$dump_directory/${job_name}_$min_func_name

    mkdir -p $final_dump_directory
    cd $final_dump_directory
    for seed in $(seq 1 8)
    do
        spython --conda_env=$env --OMP_NUM_THREADS=$CPUs --CPUs=$CPUs $py_script ${job_name}_$seed $cur_directory/$min_func_name.pkl $min_func_name $seed
    done
    cd $cur_directory
done
