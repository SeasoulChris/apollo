#!/usr/bin/env bash

export PYTHONPATH="/apollo/py_proto:/apollo/modules/data/fuel:${PYTHONPATH}"

mkdir -p /apollo/modules/data/fuel/deploy

cd /home/bae/app

# Start k8s proxy.
kubectl proxy &

# Start web server.
if [ "$1" = "--debug" ]; then
  python3 index.zip $@
else
  python3 /home/bae/http_to_https.py &
  python3 index.zip --kube_proxy='localhost' $@ > /home/bae/log/gunicorn.log 2>&1
fi
