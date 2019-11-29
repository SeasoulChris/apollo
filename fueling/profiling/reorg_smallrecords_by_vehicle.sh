#!/usr/bin/env bash

# Go to apollo-fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/../.."

SUBMITTER="./tools/submit-job-to-k8s.py --disk=1 --memory=1"
INPUT_DATA_PATH="small-records/2019"

JOB="fueling/profiling/reorg_smallrecords_by_vehicle.py"
FLAGS="--input_data_path=${INPUT_DATA_PATH}"
${SUBMITTER} --main=${JOB} --node_selector=GPU --flags="${FLAGS}"
