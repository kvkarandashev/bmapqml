#!/bin/bash
export OMP_NUM_THREADS=1
# A test_minfunc.pkl example can be created using ../minfunc_creation
python MC_opt_test.py test_minfunc.pkl
