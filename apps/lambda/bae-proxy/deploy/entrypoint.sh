#!/usr/bin/env bash

source /usr/local/miniconda/bin/activate fuel-py37
export PYTHONPATH="/apollo/modules/data/fuel:${PYTHONPATH}"

mkdir -p /apollo/modules/data/fuel/deploy /home/bae/log

cd /home/bae/app
if [ "$1" = "--debug" ]; then
  python index.py $@
else
  python index.py $@ > /home/bae/log/$(date +%Y%m%d_%H%M).log 2>&1
fi
