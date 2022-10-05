#!/bin/bash

NPROCS=20

for num_attempts in 0 1 2 4 8
do
    spython --CPUs=$NPROCS QM9_rmg_minimized_function.py rmg_min_func_$num_attempts $num_attempts $NPROCS
done
