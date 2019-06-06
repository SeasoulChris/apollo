#!/usr/bin/env bash

# Go to apollo-fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/../../.."

set -e

JOB_ID=$(date +%Y-%m-%d-%H-%M-%S)

# Feature extraction.
JOB="fueling/control/calibration_table/multi-job-feature-extraction.py"
ENV="fuel-py27-cyber"
INPUT_DATA_PATH="modules/control/data/records"
# INPUT_DATA_PATH="modules/control/calibration_table/records"
./tools/submit-job-to-k8s.sh --env ${ENV} --workers 5 --cpu 2 --memory 60g ${JOB} \
--input_data_path="${INPUT_DATA_PATH}" --job_id="${JOB_ID}"


# # Training.
# JOB="fueling/control/calibration_table/multi-vehicle-calibration-table-training.py"
# ENV="fuel-py27"
# ./tools/submit-job-to-k8s.sh --env ${ENV} --workers 5 --cpu 10 --memory 100g ${JOB} \
# --job_id="${JOB_ID}"


# # Result visualization
# JOB="fueling/control/calibration_table/multi-vehicle-calibration-result-visualization.py"
# ENV="fuel-py27"
# ./tools/submit-job-to-k8s.sh --env ${ENV} --workers 5 --cpu 2 --memory 20g ${JOB} \
# --job_id="${JOB_ID}"


# # Data distribution visualization
# JOB="fueling/control/calibration_table/multi-vehicle-data-visualization.py"
# ENV="fuel-py27"
# ./tools/submit-job-to-k8s.sh --env ${ENV} --workers 5 --cpu 2 --memory 20g ${JOB} \
# --job_id="${JOB_ID}"
