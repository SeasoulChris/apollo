#!/usr/bin/env bash

# Go to apollo-fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/../../.."

set -e

# Feature extraction.
JOB="fueling/control/calibration_table/multi-jobs-feature-extraction.py"
ENV="fuel-py27-cyber"
INPUT_DATA_PATH="modules/control/data/records"
./tools/submit-job-to-k8s.sh --env ${ENV} --workers 9 --cpu 2 --memory 20g ${JOB} \
--input_data_path="${INPUT_DATA_PATH}" --job_id="003"