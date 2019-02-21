#!/usr/bin/env bash

###############################################################################
# Copyright 2019 The Apollo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################
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
