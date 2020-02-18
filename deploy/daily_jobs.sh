#!/usr/bin/env bash

# Jobs that run daily.
# Crontab config: @daily /this/script.sh

cd "$( dirname "${BASH_SOURCE[0]}" )"

# only for internal records control profiing and visualization
INPUT_DATA_PATH="modules/control/small-records"

bazel run //deploy:daily_jobs -- --cloud \
    --workers=10 --memory=24 --disk=800 \
    --input_data_path=${INPUT_DATA_PATH}
