#!/bin/bash

# The minimized function file is created by scripts in examples/chemxpl/minfunc_creation/morfeus_xTB_solvation.
dump_directory="/store/konst/chemxpl_related/minimization_runs_xTB_dipole_solvation"

gap_constraints=("weak" "strong")

quantities=("solvation_energy" "dipole" "atomization_energy")

bias_strengths=("none" "weak" "stronger")

job_name=$1

if [ "$job_name" == "" ]
then
    echo "Job name should be second argument"
    exit
fi

cur_directory=$(pwd)

py_script=$cur_directory/MC_opt_3.py


for bias_strength in ${bias_strengths[@]}
do
    for gap_constraint in ${gap_constraints[@]}
    do
        for quantity in ${quantities[@]}
        do
            cur_job_name=${job_name}_${bias_strength}_${gap_constraint}_${quantity}
            min_func_file=$cur_directory/morfeus_xTB_water_${quantity}_${gap_constraint}.pkl
            final_dump_directory=$dump_directory/${cur_job_name}

            mkdir -p $final_dump_directory
            cd $final_dump_directory
            for seed in $(seq 1 8)
            do
                spython --OMP_NUM_THREADS=1 --CPUs=1 $py_script ${cur_job_name}_$seed $min_func_file min_$quantity $seed $bias_strength
            done
            cd $cur_directory
        done
    done
done
