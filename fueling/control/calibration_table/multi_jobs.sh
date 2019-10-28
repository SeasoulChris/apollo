#!/usr/bin/env bash

# Go to apollo-fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/../../.."

set -e

SUBMITTER="./tools/submit-job-to-k8s.py --workers=5 --cpu=5 --memory=60"
JOB_ID=$(date +%Y-%m-%d-%H-%M)
INPUT_DATA_PATH="modules/control/apollo_calibration_table"

# Feature extraction.
JOB="fueling/control/calibration_table/multi_job_feature_extraction.py"
ENV="fuel-py27-cyber"
FLAGS="--job_id=${JOB_ID} --input_data_path=${INPUT_DATA_PATH}"
${SUBMITTER} --main=${JOB} --env=${ENV} --flags="${FLAGS}"

# Training, result visualization and distribution.
JOB="fueling/control/calibration_table/multi_job_train_vis_dist.py"
ENV="fuel-py36"
FLAGS="--job_id=${JOB_ID}"
${SUBMITTER} --main=${JOB} --env=${ENV} --flags="${FLAGS}"
