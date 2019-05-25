#!/usr/bin/env bash

source /usr/local/miniconda/bin/activate fuel-py37
export PYTHONPATH="/apollo/modules/data/fuel:${PYTHONPATH}"

mkdir -p /apollo/modules/data/fuel/deploy /home/bae/log

cd /home/bae/app
# Waiting for code package.
while [ ! -f index.py ]; do
  sleep 1
done

gunicorn --reload \
    --workers=3 \
    --bind="0.0.0.0:8080" \
    --pythonpath="/apollo/modules/data/fuel" \
    --certfile="/home/bae/app/ssl_keys/cert.pem" \
    --keyfile="/home/bae/app/ssl_keys/key.pem" \
    index:app > /home/bae/log/$(date +%Y%m%d_%H%M).log 2>&1
