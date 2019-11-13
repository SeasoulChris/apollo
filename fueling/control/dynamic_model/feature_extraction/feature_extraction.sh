#!/usr/bin/env bash

# Go to apollo-fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/../../../.."

set -e
JOB_ID=$(date +%Y-%m-%d-%H)

SUBMITTER="./tools/submit-job-to-k8s.py --workers=5 --cpu=2 --memory=60 --wait"

# Feature extraction.
JOB="fueling/control/dynamic_model/feature_extraction/sample_set.py"
INPUT_DATA_PATH="modules/control/data/records"
${SUBMITTER} --main=${JOB} --flags="--input_data_path=${INPUT_DATA_PATH} --job_id=${JOB_ID}"

# Feature extraction.
JOB="fueling/control/dynamic_model/feature_extraction/uniform_set.py"
INPUT_DATA_PATH="modules/control/data/records"
${SUBMITTER} --main=${JOB} --flags="--job_id=${JOB_ID}"
