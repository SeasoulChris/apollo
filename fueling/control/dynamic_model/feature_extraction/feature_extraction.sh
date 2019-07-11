#!/usr/bin/env bash

# Go to apollo-fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/../../../.."

set -e
JOB_ID=$(date +%Y-%m-%d-%H)

# Feature extraction.
JOB="fueling/control/dynamic_model/feature_extraction/sample-set.py"
ENV="fuel-py27-cyber"
INPUT_DATA_PATH="modules/control/data/records"
./tools/submit-job-to-k8s.sh --env ${ENV} --workers 5 --cpu 2 --memory 60g ${JOB} \
--input_data_path="${INPUT_DATA_PATH}" --job_id="${JOB_ID}"


# Feature extraction.
JOB="fueling/control/dynamic_model/feature_extraction/uniform-set.py"
ENV="fuel-py27-cyber"
INPUT_DATA_PATH="modules/control/data/records"
./tools/submit-job-to-k8s.sh --env ${ENV} --workers 5 --cpu 2 --memory 60g ${JOB} \
--job_id="${JOB_ID}"
