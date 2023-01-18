#!/bin/bash

chemgraph_strings=('15#2@1@2:15#1@3:15@3:6' '15#3@1@3:15#2@2:15@3:6'
                    '16#3@1@2@3:16@2@3:6#1:6' '6#3@1:6#1@2@3:15#2@4:6#2@4:15#2'
                    '6#2@1@2:6#2@3:16#2@3:16' '6#2@1@2:6#2@3:16#2@3:16#2'
                    '6#2@1@2:6#2@3:16#1@3:16#1' '6#2@1@2:6#1@3@4:16#1@3:16#1:6#3'
                    '6#2@1@2:6#2@3:16#1:16#1' '6#2@1@2:6#2@3:16#1@3:16@4:6#3')

for s in 1 2 3
do
    str_id=0
    for str in ${chemgraph_strings[@]}
    do
        str_id=$((str_id+1))
        job_name=detailed_balance_special_valence_${str_id}_$s
        spython --CPUs=1 ../check_detailed_balance_single_moves.py $job_name $str $s
    done
done
