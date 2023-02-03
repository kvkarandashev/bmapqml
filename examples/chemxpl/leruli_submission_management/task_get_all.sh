#!/bin/bash
# A script that checks whether folders created by lpython correspond to completed jobs.
#
# If the job is completed:
# 1. downloads output ("task-get");
# 2. deletes it from wherever it was stored ("task-prune")
# The script automatically skips jobs whose output had already been downloaded.

for d in data_*
do
    echo $d
    # joblog.txt is created by leruli task-get; presence indicates bucket was already downloaded
    if [ -f $d/joblog.txt ]
    then
        continue
    fi
    cd $d
    s=$(leruli task-status)
    echo $s
    if [ "$s" != "running" ] && [ "$s" != "submitted" ]
    then
        leruli task-get
        leruli task-prune
    fi
    cd ..
done

