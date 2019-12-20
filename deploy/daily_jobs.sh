#!/usr/bin/env bash

# Jobs that run daily.
# Crontab config: @daily /this/script.sh

# Preapre: Goto fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/.."

# only for internal records control profiing and visualization
INPUT_DATA_PATH="modules/control/small-records"
FLAGS="--input_data_path=${INPUT_DATA_PATH}"

# Job: Daily jobs.
SUBMITTER="./tools/submit-job-to-k8s.py --workers=10 --memory=24"
${SUBMITTER} --main="fueling/daily_jobs.py" --disk=800 --flags="${FLAGS}"
