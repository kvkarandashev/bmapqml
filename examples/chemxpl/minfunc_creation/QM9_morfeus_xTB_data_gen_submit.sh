#!/bin/bash

NPROCS=32

for solvent in water ether acetonitrile dmso thf
do
    spython --OMP_NUM_THREADS=1 --CPU=$NPROCS QM9_morfeus_xTB_data_gen.py qm9_morfeus_xTB_solv_$solvent $solvent $NPROCS 
done
