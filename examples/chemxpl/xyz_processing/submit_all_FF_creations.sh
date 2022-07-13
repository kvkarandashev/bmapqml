#!/bin/bash
# A script that submits jobs creating MMFF and UFF xyz files for filtered QM9 xyzs.

QM9_filtered_dir=/data/konst/QM9_filtered/xyzs

for ff_type in UFF MMFF
do
    spython --CPUs=1 --OMP_NUM_THREADS=1 create_FF_coordinates.py create_${ff_type}_coordinates $QM9_filtered_dir $ff_type
done