#!/usr/bin/env bash

# Preapre: Goto fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/.."

EXECUTOR_CORES=$(cat /proc/cpuinfo | grep ^processor | wc -l)
SUBMITTER="./tools/submit-job-to-local.sh -c ${EXECUTOR_CORES}"

# Jobs to test.
set -e  # Fail on error

conda env update --prune -f conda/py36.yaml
PY36_SUBMITTER="${SUBMITTER} -e fuel-py36"
${PY36_SUBMITTER} fueling/data/pipelines/index_records.py
${PY36_SUBMITTER} fueling/data/pipelines/reorg_small_records.py
${PY36_SUBMITTER} fueling/data/pipelines/generate_small_records.py
# Add your job here.


#conda env update --prune -f conda/py36-pyro.yaml
PYRO_SUBMITTER="${SUBMITTER} -e fuel-py36-pyro"
# Add your job here.


echo "====================================================="
echo "All tests passed!"
