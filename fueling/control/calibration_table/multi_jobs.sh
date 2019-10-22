#!/usr/bin/env bash

# Go to apollo-fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/../../.."

set -e

SUBMITTER="./tools/submit-job-to-k8s.py --worker_count=5 --worker_cpu=5 --worker_memory=60"
JOB_ID=$(date +%Y-%m-%d-%H-%M)
INPUT_DATA_PATH="modules/control/apollo_calibration_table"

# Feature extraction.
JOB="fueling/control/calibration_table/multi_job_feature_extraction.py"
ENV="fuel-py27-cyber"
${SUBMITTER} --conda_env=${ENV} --entrypoint=${JOB} \
    --job_flags="--job_id=${JOB_ID} --input_data_path=${INPUT_DATA_PATH}"

# Training, result visualization and distribution.
JOB="fueling/control/calibration_table/multi_job_train_vis_dist.py"
ENV="fuel-py36"
${SUBMITTER} --conda_env=${ENV} --entrypoint=${JOB} \
    --job_flags="--job_id=${JOB_ID}"
