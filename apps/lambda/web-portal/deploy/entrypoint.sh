#!/usr/bin/env bash

export PYTHONPATH="/apollo/py_proto:/apollo/modules/data/fuel:${PYTHONPATH}"

mkdir -p /apollo/modules/data/fuel/deploy

cd /home/bae/app

if [ "$1" = "--debug" ]; then
  python3 index.py $@
else
  python3 index.py $@ > /home/bae/log/gunicorn.log 2>&1
fi
