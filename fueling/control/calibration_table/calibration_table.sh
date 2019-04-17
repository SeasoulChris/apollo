#!/usr/bin/env bash

# Go to apollo-fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/../../.."

set -e
set -x

# Make sure you are calling run_prod() instead of run_test()!
# Feature extraction.
JOB="fueling/control/calibration_table/calibration-table-feature-extraction.py"
ENV="fuel-py27-cyber"
./tools/submit-job-to-k8s.sh ${JOB} --env ${ENV} --workers 16 --cpu 2 --memory 20g

# Training.
JOB="fueling/control/calibration_table/calibration-table-training.py"
ENV="fuel-py27"
./tools/submit-job-to-k8s.sh ${JOB} --env ${ENV} --workers 1 --cpu 20 --memory 200g
