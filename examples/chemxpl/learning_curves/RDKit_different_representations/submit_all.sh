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
        for radius_parameter in '4' '8'
        do
            for useFeatures in 'True' 'False'
            do
                jobname=${jobarr_name}_${quant_name}_${kernel_type}_${radius_parameter}_${useFeatures}
                echo $jobname $quant_name $kernel_type $radius_parameter $useFeatures
            done
        done
    done
done | spython --pipe_args --OMP_NUM_THREADS=20 --CPUs=20 RDKit_FPs.py
