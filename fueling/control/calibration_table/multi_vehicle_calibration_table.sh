#!/usr/bin/env bash

# Go to apollo-fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/../../.."

set -e

# Feature extraction.
JOB="fueling/control/calibration_table/multi-vehicle-feature-extraction.py"
ENV="fuel-py27-cyber"
INPUT_DATA_PATH="modules/control/calibration_table/records"
./tools/submit-job-to-k8s.sh --env ${ENV} --workers 9 --cpu 2 --memory 20g ${JOB} \
--input_data_path="${INPUT_DATA_PATH}" --job_id="001"


# # Training.
# JOB="fueling/control/calibration_table/multi-vehicle-calibration-table-training.py"
# ENV="fuel-py27"
# INPUT_DATA_PATH="modules/control/calibration_table/records"
# ./tools/submit-job-to-k8s.sh --env ${ENV} --workers 5 --cpu 10 --memory 100g ${JOB} \
# --input_data_path="${INPUT_DATA_PATH}" --job_id="001"


# # Result visualization
# JOB="fueling/control/calibration_table/multi-vehicle-calibration-result-visualization.py"
# ENV="fuel-py27"
# INPUT_DATA_PATH="modules/control/calibration_table/records"
# ./tools/submit-job-to-k8s.sh --env ${ENV} --workers 5 --cpu 2 --memory 20g ${JOB} \
# --input_data_path="${INPUT_DATA_PATH}" --job_id="001"


# # Data distribution visualization
# JOB="fueling/control/calibration_table/multi-vehicle-data-visualization.py"
# ENV="fuel-py27"
# INPUT_DATA_PATH="modules/control/calibration_table/records"
# ./tools/submit-job-to-k8s.sh --env ${ENV} --workers 5 --cpu 2 --memory 20g ${JOB} \
# --input_data_path="${INPUT_DATA_PATH}" --job_id="001"
