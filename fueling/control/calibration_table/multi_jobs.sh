#!/usr/bin/env bash

# Go to apollo-fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/../../.."

SUBMITTER="./tools/submit-job-to-k8s.py --workers=6 --cpu=4 --memory=60"
INPUT_DATA_PATH="modules/control/apollo_calibration_table"

JOB="fueling/control/calibration_table/vehicle_calibration.py"
FLAGS="--input_data_path=${INPUT_DATA_PATH}"
${SUBMITTER} --main=${JOB} --flags="${FLAGS}"
