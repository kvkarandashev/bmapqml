#!/bin/bash

NUM_SEEDS=64

JOB_NAME=$1

RESTART_FILE=$2

if [ "$RESTART_FILE" == "" ]
then
    echo State job name and restart file
    exit
fi

if [ "${RESTART_FILE::1}" != "/" ]
then
    RESTART_FILE=$(pwd)/$RESTART_FILE
fi

dir=$JOB_NAME
mkdir $dir
cd $dir

for SEED in $(seq $NUM_SEEDS)
do
    spython --CPUs=1 ../check_detailed_balance_single_moves_from_restart.py ${JOB_NAME}_$SEED $RESTART_FILE $SEED
done
