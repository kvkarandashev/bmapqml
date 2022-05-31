#!/bin/bash

# Check that the directory with qm7 XYZ's is unpacked.

if [ ! -d qm7 ]
then
    tar -xf ../test_data/qm7.tar.gz
fi

python ../../clear_cache.py

# Which tests to write.

#used_tests=(pair_Gauss_oxidation_molpro FJK_pair_sep_ibo_kernel_spin_change_UHF FJK_sep_Gauss_kernel_wders\
#            FJK_pair_sep_global_Gauss_kernel_wders FBCM_sep_Gauss_kernel_wders_xTB OFD_sep_Gauss_kernel_wders_xTB)

used_tests=( FJK_pair_sep_global_Gauss_kernel_wders )

export OMP_NUM_THREADS=2
export BMAPQML_NUM_PROCS=2 # The variable says how many processes to use during joblib-parallelized parts.
                        # Note that learning_curve_building.py at init_compound_list uses a dirty hack to assign it equalling OMP_NUM_THREADS.

for test_script in ${used_tests[@]} 
do
	python $test_script.py $test_script.log
	bench=benchmark_data/$test_script.dat
	if [ -f $bench ]
	then
		diff_lines=$(diff $test_script.log $bench | wc -l)
		if [ "$diff_lines" == "0" ]
		then
			echo "Passed: $test_script"
		else
			echo "Possible Failure: $test_script"
		fi
	else
		mv $test_script.log $bench
		echo "Benchmark created: $test_script"
	fi
done
