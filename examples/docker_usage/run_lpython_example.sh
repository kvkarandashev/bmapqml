#!/bin/bash
lpython --CPUs=2 --memory=20000 --update_bmapqml --docker=bmapqml:1.0 --req_files=test_minfunc.pkl MC_opt_test.py sample_job test_minfunc.pkl
