#!/usr/bin/env bash

# Go to apollo-fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/../../../.."

set -e
JOB_ID=$(date +%Y-%m-%d-%H)

SUBMITTER="./tools/submit-job-to-k8s.py --conda_env=fuel-py27-cyber \
    --worker_count=5 --worker_cpu=2 --worker_memory=60"

# Feature extraction.
JOB="fueling/control/dynamic_model/feature_extraction/sample_set.py"
INPUT_DATA_PATH="modules/control/data/records"
${SUBMITTER} --entrypoint=${JOB} --job_flags="--input_data_path=${INPUT_DATA_PATH} --job_id=${JOB_ID}"


# Feature extraction.
JOB="fueling/control/dynamic_model/feature_extraction/uniform_set.py"
INPUT_DATA_PATH="modules/control/data/records"
${SUBMITTER} --entrypoint=${JOB} --job_flags="--job_id=${JOB_ID}"
