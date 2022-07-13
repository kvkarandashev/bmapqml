#!/bin/bash

if [ "$1" == "" ]
then
    echo "Job array name should be second argument"
    exit
else
    jobarr_name=$1
fi

for quant_name in 'Dipole_moment' 'HOMO-LUMO_gap' #'Atomization_energy' 'Normalized_atomization_energy'
do
    for kernel_type in 'Gaussian' 'Laplacian'
    do
        for ff_type in 'UFF' 'MMFF'
        do
            jobname=${jobarr_name}_${quant_name}_${kernel_type}_${ff_type}
            echo $jobname $quant_name $kernel_type $ff_type
        done
    done
done  | spython --pipe_args --OMP_NUM_THREADS=20 --CPUs=20 SLATM_FF_combination_lc_var_kernel.py
