#!/usr/bin/env bash

bazel run //fueling/control/calibration_table:vehicle_calibration -- \
    --cloud --workers=6 --cpu=4 --memory=60 \
    --input_data_path="modules/control/apollo_calibration_table"
