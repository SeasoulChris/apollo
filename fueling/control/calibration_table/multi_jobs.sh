#!/usr/bin/env bash

# Go to apollo-fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/../../.."

set -e
JOB_ID=$(date +%Y-%m-%d-%H)

# Feature extraction.
JOB="fueling/control/calibration_table/multi-job-feature-extraction.py"
ENV="fuel-py27-cyber"
INPUT_DATA_PATH="modules/control/apollo_calibration_table"
# INPUT_DATA_PATH="modules/control/data/records"
# INPUT_DATA_PATH="modules/control/calibration_table/records"
./tools/submit-job-to-k8s.sh --env ${ENV} --workers 5 --cpu 2 --memory 60g ${JOB} \
--input_data_path="${INPUT_DATA_PATH}" --job_id="${JOB_ID}"


# Training.
JOB="fueling/control/calibration_table/multi-job-train.py"
ENV="fuel-py27"
./tools/submit-job-to-k8s.sh --env ${ENV} --workers 5 --cpu 5 --memory 60g ${JOB} \
--job_id="${JOB_ID}"


# Result visualization
JOB="fueling/control/calibration_table/multi-job-result-visualization.py"
ENV="fuel-py27"
./tools/submit-job-to-k8s.sh --env ${ENV} --workers 5 --cpu 2 --memory 20g ${JOB} \
--job_id="${JOB_ID}"


# Data distribution visualization
JOB="fueling/control/calibration_table/multi-job-data-distribution.py"
ENV="fuel-py27"
./tools/submit-job-to-k8s.sh --env ${ENV} --workers 5 --cpu 2 --memory 20g ${JOB} \
--job_id="${JOB_ID}"
