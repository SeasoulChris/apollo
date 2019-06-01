#!/usr/bin/env bash

source /usr/local/miniconda/bin/activate fuel-py37
export PYTHONPATH="/apollo/py_proto:/apollo/modules/data/fuel:${PYTHONPATH}"

mkdir -p /apollo/modules/data/fuel/deploy

cd /home/bae/app

if [ "$1" = "--debug" ]; then
  python index.py $@
else
  python index.py $@ > /home/bae/log/gunicorn.log 2>&1
fi
