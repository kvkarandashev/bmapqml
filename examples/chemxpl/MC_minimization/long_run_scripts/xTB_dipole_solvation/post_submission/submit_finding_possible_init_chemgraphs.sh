#!/bin/bash

MYDIR=$(pwd)

for dataset in qm9 egp
do
    if [ "$dataset" == "qm9" ]
    then
        xyz_dir=QM9_formatted_xyzs
    else
        xyz_dir=EGP/new_xyzs
    fi 
    quant_func_pkl=$MYDIR/../${dataset}_morfeus_xTB_cheap_water_solvation_energy_strong.pkl

    jobname=init_chemgraphs_$dataset

    full_xyz_dir=/data/konst/$xyz_dir
    spython --CPUs=40 --OMP_NUM_THREADS=1 find_possible_init_chemgraphs.py $jobname $full_xyz_dir $dataset $quant_func_pkl
done
