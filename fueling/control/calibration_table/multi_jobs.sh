#!/usr/bin/env bash

# Go to apollo-fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/../../.."

set -e

SUBMITTER="./tools/submit-job-to-k8s.sh --workers 5 --cpu 5 --memory 60g"
JOB_ID=$(date +%Y-%m-%d-%H)
INPUT_DATA_PATH="modules/control/apollo_calibration_table"

# Feature extraction.
JOB="fueling/control/calibration_table/multi_job_feature_extraction.py"
ENV="fuel-py27-cyber"
${SUBMITTER} --env ${ENV} ${JOB} --job_id="${JOB_ID}" --input_data_path="${INPUT_DATA_PATH}"

# Training, result visualization and distribution.
JOB="fueling/control/calibration_table/multi_job_train_vis_dist.py"
ENV="fuel-py36"
${SUBMITTER} --env ${ENV} ${JOB} --job_id="${JOB_ID}"
