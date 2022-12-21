#!/bin/bash
#Use bmapqml:1.0 docker image (created using scripts in example/docker_creation)
docker run -v "$(pwd):/rundir" bmapqml:1.0 python test_env.py
docker run -v "$(pwd):/rundir" bmapqml:1.0 ./adjusted_MC_opt_test_run.sh
