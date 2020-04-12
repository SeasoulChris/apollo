#!/usr/bin/env bash

export PYTHONPATH=/home/afs_data_service:$PYTHONPATH
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./afs_data_service.proto
source /home/afs_data_service/adbsdk/setup.bash

python server.py 
