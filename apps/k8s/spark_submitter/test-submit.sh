#!/usr/bin/env bash

curl http://localhost:8000/ -X POST -H "Content-Type: application/json" -d \
'{
  "env": {"node_selector": "GPU"},
  "job": {"entrypoint": "fueling/demo/gpu_training_with_pytorch.py"}
}'
