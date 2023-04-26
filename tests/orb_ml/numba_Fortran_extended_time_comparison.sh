#!/bin/bash

export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

# Check that the directory with qm7 XYZ's is unpacked.

if [ ! -d qm7 ]
then
    tar -xf ../test_data/qm7.tar.gz
fi

# Note that the *.log files are identical, verifying that the three implementations do the same thing.
for implementation in numba numba_types Fortran
do
    echo "Implementation tested: $implementation"
    python FJK_test_implementation.py kernels_time_test_$implementation.log $implementation
done
