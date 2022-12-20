#!/bin/bash

NPROCS=40
CPUS=1

for solvent in water ether #acetonitrile dmso thf
do
    spython --OMP_NUM_THREADS=$CPUS --CPUs=$NPROCS QM9_morfeus_xTB_data_gen_no_cut.py qm9_morfeus_xTB_solv_no_cut_$solvent $solvent $((NPROCS/CPUS))
done
