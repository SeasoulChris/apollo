#!/usr/bin/env bash

# Jobs that run once every week.
# Crontab example: @weekly /this/script.sh

set -e

# Preapre: Goto fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )"

# Job: Prediction data labeling.
# bazel run //fueling/prediction:prediction_app_data_labeling -- --cloud \
#    --workers=10 --memory=30 --disk=50

# Job: Prediction performance evaluation.
# bazel run //fueling/prediction:prediction_app_performance_evaluation -- --cloud \
#    --workers=10 --memory=30 --disk=200
