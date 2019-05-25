#!/usr/bin/env bash

source /usr/local/miniconda/bin/activate fuel-py37
export PYTHONPATH="/apollo/modules/data/fuel:${PYTHONPATH}"

mkdir -p /apollo/modules/data/fuel/deploy

cd /home/bae/app
gunicorn --reload \
    --workers=3 \
    --bind="0.0.0.0:8080" \
    --pythonpath="/apollo/modules/data/fuel" \
    --certfile="ssl_keys/cert.pem" \
    --keyfile="ssl_keys/key.pem" \
    index:app > /home/bae/log/gunicorn.log 2>&1
