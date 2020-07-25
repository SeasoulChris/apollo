#!/usr/bin/env bash

INPUT_DATA_PATH="test/simplehdmap"
ZONE_ID=50
LIDAR_TYPE="velodyne16"
LANE_WIDTH=3.0
EXTRA_ROI_EXTENSION=1.0

bazel run //fueling/map:generate_maps -- --cloud --workers=5 --cpu=5 --memory=60 \
    --input_data_path=${INPUT_DATA_PATH} --zone_id=$ZONE_ID --lidar_type=$LIDAR_TYPE \
    --lane_width=$LANE_WIDTH --extra_roi_extension=$EXTRA_ROI_EXTENSION
