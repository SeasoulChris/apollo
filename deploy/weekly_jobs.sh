#!/usr/bin/env bash

# Jobs that run once every week, starting at 6:00 a.m. on Saturday.
# Crontab example: 0 6 * * 6 /this/script.sh

set -e

# Preapre: Goto fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/.."

# Job: Prediction data labeling.
# JOB="fueling/prediction/prediction_app_data_labeling.py"
# ./tools/submit-job-to-k8s.py --worker_count=10 --worker_memory=30 --worker_disk=50 \
#     --entrypoint=${JOB}

# Job: Prediction performance evaluation.
# JOB="fueling/prediction/prediction_app_performance_evaluation.py"
# ./tools/submit-job-to-k8s.py --worker_count=10 --worker_memory=30 --worker_disk=200 \
#     --entrypoint=${JOB}
