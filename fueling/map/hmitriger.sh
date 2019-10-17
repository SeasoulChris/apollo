#!/usr/bin/env bash

# Go to apollo-fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/../.."

set -e

SUBMITTER="./tools/submit-job-to-k8s.sh --workers 5 --cpu 5 --memory 60g"
JOB_ID=$(date +%Y-%m-%d-%H)
INPUT_DATA_PATH="simplehdmap"

# Job: map generate base_map.txt
JOB="fueling/map/generate_base_map.py"
ENV="fuel-py27-cyber"
${SUBMITTER} --env ${ENV} ${JOB} --job_id="${JOB_ID}" --input_data_path="${INPUT_DATA_PATH}"

# Job: map generate local_map.txt
JOB="fueling/map/generate_local_map.py"
ENV="fuel-py27-cyber"
${SUBMITTER} --env ${ENV} ${JOB} --job_id="${JOB_ID}" --input_data_path="${INPUT_DATA_PATH}"

# Job: map generate for sim_map.txt or routing_map.txt
JOB="fueling/map/generate_sim_routing_map.py"
ENV="fuel-py27-cyber"
${SUBMITTER} --env ${ENV} ${JOB} --job_id="${JOB_ID}" --input_data_path="${INPUT_DATA_PATH}"
