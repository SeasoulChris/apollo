#!/usr/bin/env bash
#
# Usage:
# sudo bash /apollo/modules/tools/prediction/map_feature/generate_img.sh <input_dir> <output_dir>
#
# The input frame_env.X.bin will generate imgs: obs_id@timestamp.png and obs_pos.npy

SRC_DIR=$1
TARGET_DIR=$2
REGION=$3

set -e

source /apollo/scripts/apollo_base.sh
source /apollo/cyber/setup.bash 

python modules/tools/prediction/map_feature/generate_img.py -i=${SRC_DIR} -o=${TARGET_DIR} -r=${REGION}
