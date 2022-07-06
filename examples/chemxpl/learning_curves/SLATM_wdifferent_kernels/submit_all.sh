#!/bin/bash

if [ "$1" == "" ]
then
    echo "Job array name should be second argument"
    exit
else
    jobarr_name=$1
fi

for quant_name in 'Dipole_moment' 'HOMO-LUMO_gap' 'Atomization_energy' 'Normalized_atomization_energy'
do
    for kernel_type in 'Gaussian' 'Laplacian'
    do
        jobname=${jobarr_name}_${quant_name}_${kernel_type}
        spython SLATM_MMFF_combination_lc_var_kernel.py $jobname $quant_name $kernel_type
    done
done
