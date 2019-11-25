#!/usr/bin/env bash

# Go to apollo-fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/../.."

SUBMITTER="./tools/submit-job-to-k8s.py --workers=6 --cpu=4 --memory=60"
INPUT_DATA_PATH="modules/control/profiling/multi_job"

JOB="fueling/profiling/multi_job_control_profiling_metrics.py"
FLAGS="--input_data_path=${INPUT_DATA_PATH}"
${SUBMITTER} --main=${JOB} --flags="${FLAGS}"
