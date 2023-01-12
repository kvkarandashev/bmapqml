#!/bin/bash

submitter="lpython" # spython?

dumpdir=MC_graph_enum

mkdir -p $dumpdir

cd $dumpdir

for IMPLICIT_CONSTRAINT in TRUE FALSE
do
    for bias_type in "none" "weak" "stronger"
    do 
        for nhatoms in $(seq 3 6)
        do
            for USE_GENETIC in TRUE FALSE
            do
                for seed in $(seq 1 8)
                do
                    jobname=job_${nhatoms}_${bias_type}_${USE_GENETIC}_${IMPLICIT_CONSTRAINT}_${seed}
                    if [ "$USE_GENETIC" == "TRUE" ] && [ "$nhatoms" == "3" ] # genetic steps won't work in this situation.
                    then
                        continue
                    fi
                    if [ "$submitter" == "lpython" ]
                    then
                        lpython --update_bmapqml --memory=10000 ../MC_graph_enumeration.py $jobname $seed $nhatoms $IMPLICIT_CONSTRAINT $bias_type $USE_GENETIC
                    fi
                done
            done
        done
    done
done