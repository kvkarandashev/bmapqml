#!/bin/bash

# Replace with wherever you want to store the results.
dump_directory="/store/common/konst/chemxpl_related/minimization_runs_xTB"

min_func_directory="/store/common/konst/chemxpl_related"

min_func_names=("xTB_MMFF94_morfeus_electrolyte")

job_name=$1

if [ "$job_name" == "" ]
then
    echo "Job name should be second argument"
    exit
fi

cur_directory=$(pwd)

py_script=$cur_directory/MC_no_bias_4.py


for min_func_name in ${min_func_names[@]}
do
    final_dump_directory=$dump_directory/${job_name}_$min_func_name

    mkdir -p $final_dump_directory
    cd $final_dump_directory
    for seed in $(seq 1 8)
    do
        spython --OMP_NUM_THREADS=2 --CPUs=2 $py_script ${job_name}_$seed $min_func_directory/minimized_function_${min_func_name}.pkl $min_func_name $seed
    done
done
