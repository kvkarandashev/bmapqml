#!/bin/bash

# Check that the directory with qm7 XYZ's is unpacked.

if [ ! -d qm7 ]
then
    tar -xf ../test_data/qm7.tar.gz
fi

for use_Fortran in True False
do
    echo "FORTRAN USED: $use_Fortran"
    python FJK_pair_sep_global_Gauss_kernel_wders.py use_Fortran_$use_Fortran.log $use_Fortran
done
