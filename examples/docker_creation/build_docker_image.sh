#!/bin/bash

docker_image_name=$1

docker_spec=$2

if [ "$docker_spec" == "" ]
then
    echo "Use build image name as first argument and docker specification as second."
    exit
fi

rm -f Dockerfile

python make_docker_file.py $2

docker image build -t $1 .
