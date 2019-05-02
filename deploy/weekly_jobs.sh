#!/usr/bin/env bash

# Jobs that run once every week, starting at 6:00 a.m. on Saturday.
# Crontab example: 0 6 * * 6 /this/script.sh

set -e

# Preapre: Goto fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/.."

# Job: Prediction data labeling.
JOB="fueling/prediction/prediction-app-data-labeling.py"
./tools/submit-job-to-k8s.sh --workers 8 --memory 10g ${JOB}
