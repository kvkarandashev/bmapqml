#!/bin/bash

NPROCS=40
CPUS=2

for solvent in water ether acetonitrile dmso thf
do
    spython --OMP_NUM_THREADS=$CPUS --CPUs=$NPROCS QM9_morfeus_xTB_data_gen.py qm9_morfeus_xTB_solv_$solvent $solvent $((NPROCS/CPUS))
done
