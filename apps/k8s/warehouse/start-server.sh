#!/usr/bin/env bash

source /usr/local/miniconda2/bin/activate fuel-py36
export PYTHONPATH="/apollo/py_proto:/apollo/modules/data/fuel:${PYTHONPATH}"

cd /apollo/modules/data/fuel/apps/k8s/warehouse
nohup python main.py > log.txt 2>&1 &
