#!/bin/bash

for num_attempts in 0 1 2 4 8
do
    spython --CPUs=1 QM9_rmg_minimized_function.py rmg_min_func_$num_attempts $num_attempts 
done