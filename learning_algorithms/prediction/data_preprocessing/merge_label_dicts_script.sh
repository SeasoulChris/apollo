#!/usr/bin/env bash
#
# Usage:
# sudo bash /apollo/modules/tools/prediction/mlp_train/scripts/generate_labels.sh <input_feature.bin>
#
# The input feature.X.bin will generate furture_status.label, cruise.label, junction.label

SRC_FILE=$1

set -e

source /apollo/scripts/apollo_base.sh
source /apollo/cyber/setup.bash 

python modules/tools/prediction/learning_algorithms/data_preprocessing/merge_label_dicts.py ${SRC_FILE}
