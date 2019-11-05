#!/usr/bin/env bash

# Preapre: Goto fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/.."

EXECUTOR_CORES=$(cat /proc/cpuinfo | grep ^processor | wc -l)
SUBMITTER="./tools/submit-job-to-local.sh -c ${EXECUTOR_CORES}"

# Update Conda env.
ls conda/*.yaml | xargs -L 1 conda env update --prune -f

# Jobs to test.
set -e  # Fail on error

CONDA_ENV="fuel-py27-cyber"
${SUBMITTER} -e ${CONDA_ENV} fueling/data/pipelines/index_records.py
${SUBMITTER} -e ${CONDA_ENV} fueling/data/pipelines/reorg_small_records.py
${SUBMITTER} -e ${CONDA_ENV} fueling/data/pipelines/generate_small_records.py

echo "All tests passed!"
