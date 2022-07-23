#!/bin/bash

# Replace with wherever you want to store the results.
dump_directory="/store/common/konst/chemxpl_related/minimization_runs"

min_func_directory="/store/common/konst/chemxpl_related"

min_func_names=("electrolyte")

job_name=$1

if [ "$job_name" == "" ]
then
    echo "Job name should be second argument"
    exit
fi

cur_directory=$(pwd)

py_script=$cur_directory/MC_no_bias.py


for min_func_name in ${min_func_names[@]}
do
    final_dump_directory=$dump_directory/${job_name}_$min_func_name

    mkdir -p $final_dump_directory
    cd $final_dump_directory
    for seed in $(seq 1 8)
    do
        spython --OMP_NUM_THREADS=20 --CPUs=20 $py_script ${job_name}_$seed $min_func_directory/minimized_function_$min_func_name.pkl $min_func_name $seed
    done
done
