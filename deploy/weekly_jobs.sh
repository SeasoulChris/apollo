#!/usr/bin/env bash

# Jobs that run once every week, starting at 6:00 a.m. on Saturday.
# Crontab example: @weekly /this/script.sh

set -e

# Preapre: Goto fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/.."

# Job: Prediction data labeling.
# JOB="fueling/prediction/prediction_app_data_labeling.py"
# ./tools/submit-job-to-k8s.py --main=${JOB} --workers=10 --memory=30 --disk=50

# Job: Prediction performance evaluation.
# JOB="fueling/prediction/prediction_app_performance_evaluation.py"
# ./tools/submit-job-to-k8s.py --main=${JOB} --workers=10 --memory=30 --disk=200
