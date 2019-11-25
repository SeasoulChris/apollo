#!/usr/bin/env bash

cd $( dirname "${BASH_SOURCE[0]}" )

APOLLO=$(cd ../../../apollo; pwd)
APOLLO_FUEL=$(cd ../..; pwd)

export PYTHONPATH="${APOLLO}/py_proto:${APOLLO_FUEL}:${PYTHONPATH}"
python3 index.py --debug
