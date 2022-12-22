#!/bin/bash

docker stop $(docker ps -a -q)
docker rm $(docker ps -a -q)

image_ids=$(docker images | awk '{if ($1 == "<none>") print $3}')
for i in ${image_ids[@]}
do
    docker rmi $i
done
